import torch
import csv
from tqdm import tqdm
from datetime import datetime
from torch.amp import autocast, GradScaler
from training.callbacks import CallbackHandler
from training.metrics import calculate_metrics
from utils.other import getPromptEmbeddings
from typing import Dict, List, Tuple, Any

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        bert: Any,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        logger: Any,
        monitor: Any,
        grad_log: str,
        is_grad: bool = False
    ):
        self.model = model
        self.bert = bert
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.logger = logger
        self.is_grad = is_grad
        self.grad_log = grad_log
        
        # 混合精度训练设置
        self.mixed_precision = config['experiment']['techniques']['mixed_precision']
        self.scaler = GradScaler(enabled=self.mixed_precision)
        
        # 梯度裁剪设置
        self.gradient_clipping = config['experiment']['techniques']['gradient_clipping']
        
        # 创建回调处理器
        self.callbacks = CallbackHandler(
            model=model,
            config=config,
            logger=logger,
            device=device,
            optimizer=optimizer,
            monitor=monitor
        )
        
        if self.is_grad:
            self._init_grad_log()
                
    def train_epoch(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: torch.nn.Module
    ) -> float:
        
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training")
        step = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            self.callbacks.on_batch_begin(batch_idx)
            current_lr = optimizer.param_groups[0]['lr']
            
            # prompts = getPromptEmbeddings(
            #     batch['prompts'], 
            #     self.bert, 
            #     self.tokenizer, 
            #     self.device, 
            #     max_length=self.config['model']['max_length']
            # )
            
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'prompts': batch['prompts']
            }
            targets = batch['target'].to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', enabled=self.config['experiment']['techniques']['mixed_precision']):
                logits = self.model(**inputs)
                loss = loss_fn(logits.permute(0, 2, 1), targets.long())
            
            # 混合精度训练
            self.scaler.scale(loss).backward()
            
            # 梯度监控和记录
            if self.is_grad:
                self.monitor_gradients(threshold=1000, step=step)
                self._log_gradients_to_file(step)
            
            # 梯度裁剪（支持混合精度和非混合精度）
            if self.gradient_clipping > 0:
                if self.mixed_precision:
                    self.scaler.unscale_(optimizer)
                    
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clipping
                )
            
            self.scaler.step(optimizer)
            self.scaler.update()
            
            step += 1
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.6f}")
            self.callbacks.on_batch_end(batch_idx, loss=loss.item())
            
        return total_loss / len(train_loader)
    
    def validate(
        self, 
        val_loader: torch.utils.data.DataLoader, 
        loss_fn: torch.nn.Module
    ) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                targets = batch['target'].to(self.device)
                
                with autocast(device_type='cuda', enabled=self.mixed_precision):
                    logits = self.model(**inputs)
                    loss = loss_fn(logits.permute(0, 2, 1), targets.long())
                
                probs = torch.nn.functional.softmax(logits, dim=2)
                outputs = torch.argmax(probs, dim=2)
                
                total_loss += loss.item()
                all_outputs.append(outputs.detach().cpu())
                all_targets.extend(targets.detach().cpu())
        
        avg_loss = total_loss / len(val_loader)
        metrics = calculate_metrics(all_outputs, all_targets)
        
        return avg_loss, metrics

    def _init_grad_log(self) -> None:
        """初始化梯度日志文件并写入表头"""
        with open(self.grad_log, 'w', newline='') as f:
            writer = csv.writer(f)
            # 创建表头：Step, Timestamp, 所有参数层名称
            headers = ["Step", "Timestamp"] + [name for name, _ in self.model.named_parameters()]
            writer.writerow(headers)
    
    def _log_gradients_to_file(self, step: int) -> None:
        """将当前步骤的梯度数据写入日志文件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        scale_factor = self.scaler.get_scale() if self.mixed_precision else 1.0
        
        # 收集所有层的梯度范数
        grad_data = {"Step": step, "Timestamp": timestamp}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                actual_grad = param.grad / scale_factor
                grad_norm = actual_grad.norm().item()
                grad_data[name] = grad_norm
            elif param.grad is None:
                grad_data[name] = "None"
            else:
                grad_data[name] = "Warning"

        with open(self.grad_log, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [grad_data.get("Step", ""), grad_data.get("Timestamp", "")]
            row.extend(grad_data.get(name, "NaN") for name, _ in self.model.named_parameters())
            writer.writerow(row)
        
    def monitor_gradients(
        self, 
        step: int, 
        threshold: float = 1000.0, 
        verbose: bool = True
    ) -> None:
        """监控梯度并检测梯度爆炸
        
        Args:
            step: 当前训练步数
            threshold: 梯度爆炸的阈值（L2范数）
            verbose (bool): 是否打印警告信息
        """
        max_grad = 0
        scale_factor = self.scaler.get_scale() if self.mixed_precision else 1.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 计算该参数梯度的L2范数
                actual_grad = param.grad / scale_factor
                grad_norm = actual_grad.norm().item()
                           
                # 更新最大梯度值
                if grad_norm > max_grad:
                    max_grad = grad_norm
                
                # 检测梯度爆炸
                if grad_norm > threshold:
                    if verbose:
                        warning_msg = (
                            f"⚠️ Gradient explosion warning (Step {step}): "
                            f"Parameter '{name}' gradient norm = {grad_norm:.4e}"
                        )
                        tqdm.write(warning_msg)
                        self.logger.warning(warning_msg)

    @classmethod
    def load_grad_history(cls, log_file_path: str) -> Dict[str, List[Tuple[int, float]]]:
        """从日志文件加载梯度历史数据"""
        from collections import defaultdict
        grad_history = defaultdict(list)
        
        with open(log_file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            layer_names = headers[2:]
            
            for row in reader:
                step = int(row[0])
                for i, value in enumerate(row[2:]):
                    if value != "NaN":
                        grad_history[layer_names[i]].append((step, float(value)))
        
        return grad_history