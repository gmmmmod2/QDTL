import logging
import joblib
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from pathlib import Path
from typing import Dict, Tuple, Union

logger = logging.getLogger("data.DataLoader")

class CustomDataset(Dataset):
    """自定义数据集类，处理数据加载和预处理"""
    
    def __init__(
        self, 
        data: pd.DataFrame,
        tokenizer: BertTokenizer,
        mode: str,
        max_length: int = 128,
        seed: int = 42,
        cache_dir: str = "data/processed/data_cache",
        ):
        self.data = data
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.label_prefixes = ['Professionalism', 'Occupational', 'Effectiveness', 'Quality', 'Other']
        self.target_columns = [f"{prefix}_score" for prefix in self.label_prefixes]
        
        self._validate_data()
        self._preprocess_and_cache()
        
    def _validate_data(self):
        """验证数据集中是否包含所有必要的列"""
        required_columns = ['comments'] + \
                          [f"{prefix}_reason" for prefix in self.label_prefixes] + \
                          self.target_columns
        missing_cols = [col for col in required_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"数据集缺少必要列: {missing_cols}")
        
    def _preprocess_and_cache(self):
        """预处理数据并缓存结果"""
        cache_file = self.cache_dir / f"processed_{self.mode}_{self.seed}_{len(self.data)}_{self.max_length}.pkl"
        if cache_file.exists():
            self.cached_data = joblib.load(cache_file)
            logger.info(f"从缓存加载预处理数据: {cache_file}")
            return
            
        self.cached_data = []
        for idx in range(len(self.data)):
            try:
                text = str(self.data.iloc[idx]['comments'])
                text_encoding = self.tokenizer(
                    text, 
                    max_length=self.max_length,
                    padding='max_length', 
                    truncation=True,
                    return_tensors='pt'
                )
                
                prompts = []
                for prefix in self.label_prefixes:
                    reason_col = f"{prefix}_reason"
                    prompt = str(self.data.iloc[idx][reason_col])
                    prompts.append(prompt)
                
                target_scores = self.data[self.target_columns].iloc[idx].values
                if not all(0 <= score <= 5 for score in target_scores):
                    raise ValueError(f"索引 {idx} 的目标分数超出0-5范围: {target_scores}")
                target_tensor = torch.tensor(target_scores, dtype=torch.float)
                
                self.cached_data.append({
                    'input_ids': text_encoding['input_ids'].squeeze(0),
                    'attention_mask': text_encoding['attention_mask'].squeeze(0),
                    'prompts': prompts,
                    'target': target_tensor
                })
                
            except Exception as e:
                logger.error(f"处理索引 {idx} 时出错: {str(e)}")
                continue  # 跳过问题样本
                
        joblib.dump(self.cached_data, cache_file)
        logger.info(f"预处理数据已缓存: {cache_file}")
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        return self.cached_data[idx]

class DataLoaderModule:
    """数据加载模块"""
    
    def __init__(self, config: Dict[str, Union[str, int, float]], tokenizer: BertTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._load_data()
        self._split_datasets()
        
    def _load_data(self) -> None:
        logger.info(f"从 {self.config['csv_path']} 加载数据")
        try:
            self.full_data = pd.read_csv(self.config['csv_path'])
            if self.full_data.isnull().sum().sum() > 0:
                logger.warning("数据包含缺失值，可能会影响模型训练")
            logger.info(f"数据加载成功，总样本数: {len(self.full_data)}")
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            raise
            
    def _split_datasets(self) -> None:
        logger.info("划分数据集...")
        np.random.seed(self.config['random_seed'])
        torch.manual_seed(self.config['random_seed'])
        
        if abs(self.config['train_ratio'] + self.config['val_ratio'] + self.config['test_ratio'] - 1.0) > 1e-6:
            raise ValueError("训练集、验证集和测试集的比例之和必须为1.0")
        
        total_size = len(self.full_data)
        train_size = int(total_size * self.config['train_ratio'])
        val_size = int(total_size * self.config['val_ratio'])
        test_size = total_size - train_size - val_size
        
        if test_size < 0:
            raise ValueError("划分比例总和不能超过1.0")
        
        indices = np.arange(total_size)
        train_indices, temp_indices = train_test_split(
            indices, 
            test_size =1 - self.config['train_ratio'],
            random_state = self.config['random_seed']
        )
        
        val_test_ratio = self.config['test_ratio'] / (self.config['val_ratio'] + self.config['test_ratio'])
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size = val_test_ratio,
            random_state = self.config['random_seed']
        )
        
        self.train_dataset = CustomDataset(
            self.full_data.iloc[train_indices].reset_index(drop=True),
            self.tokenizer,
            'TRAIN',
            self.config['max_length'],
            self.config['random_seed'],
        )
        
        self.val_dataset = CustomDataset(
            self.full_data.iloc[val_indices].reset_index(drop=True),
            self.tokenizer,
            'VAL',
            self.config['max_length'],
            self.config['random_seed'],

        )
        
        self.test_dataset = CustomDataset(
            self.full_data.iloc[test_indices].reset_index(drop=True),
            self.tokenizer,
            'TEST',
            self.config['max_length'],
            self.config['random_seed'],
        )
        
        logger.info(f"数据集划分完成: 训练集={len(self.train_dataset)}, "
                    f"验证集={len(self.val_dataset)}, 测试集={len(self.test_dataset)}")
        
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=min(self.config['batch_size'], len(self.val_dataset)),
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=min(self.config['batch_size'], len(self.test_dataset)),
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
if __name__ == '__main__':
    import os
    tokenizer = BertTokenizer.from_pretrained("models/save/tokenizer")
    CONFIG = {
        'csv_path': 'data/processed/data.csv',
        'max_length': 128,
        'batch_size': 64,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        'num_workers': min(4, os.cpu_count() - 1),
    }
    data_module = DataLoaderModule(CONFIG, tokenizer)
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    def inspect_data_loader(loader, loader_name, max_batches=200):
        print(f"🔍 开始检查 {loader_name} 数据加载器，最多检查 {max_batches} 个批次...")
        
        # 存储发现的问题
        problems_found = []
        
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            print(f"\n=== 批次 {batch_idx} ===")
            
            # 1. 检查每个字段是否存在NaN/Inf
            for key in ['input_ids', 'attention_mask', 'target']:
                data = batch[key]
                
                # 检查NaN
                nan_mask = torch.isnan(data)
                if nan_mask.any():
                    nan_count = nan_mask.sum().item()
                    problems_found.append(f"{key} 包含NaN值 (数量: {nan_count})")
                    print(f"❌ 问题: {key} 包含NaN值 (数量: {nan_count})")
                
                # 检查无穷值
                inf_mask = torch.isinf(data)
                if inf_mask.any():
                    inf_count = inf_mask.sum().item()
                    problems_found.append(f"{key} 包含无穷值 (数量: {inf_count})")
                    print(f"❌ 问题: {key} 包含无穷值 (数量: {inf_count})")
            
            # 2. 检查input_ids的范围
            input_ids = batch['input_ids']
            min_id = input_ids.min().item()
            max_id = input_ids.max().item()
            print(f"input_ids 范围: {min_id} 到 {max_id}")
            
            if min_id < 0:
                problems_found.append(f"input_ids 包含负值 (最小值: {min_id})")
                print(f"❌ 问题: input_ids 包含负值 (最小值: {min_id})")
            
            # 3. 检查attention_mask的值
            attn_mask = batch['attention_mask']
            unique_vals = torch.unique(attn_mask).tolist()
            print(f"attention_mask 唯一值: {unique_vals}")
            
            if set(unique_vals) - {0, 1}:
                problems_found.append(f"attention_mask 包含非法值: {unique_vals}")
                print(f"❌ 问题: attention_mask 包含非法值: {unique_vals}")
            
            # 4. 检查target标签
            targets = batch['target']
            
            # 检查多标签分类的target值
            if targets.dim() > 1:  # 多标签情况
                min_val = targets.min().item()
                max_val = targets.max().item()
                print(f"target 值范围: {min_val} 到 {max_val}")
                
                if min_val < 0 or max_val > 5:
                    problems_found.append(f"target 值超出[0,1]范围: [{min_val}, {max_val}]")
                    print(f"❌ 问题: target 值超出[0,1]范围: [{min_val}, {max_val}]")
            
            # 5. 检查数据形状一致性
            batch_size = input_ids.size(0)
            for key in ['input_ids', 'attention_mask', 'target']:
                if batch[key].size(0) != batch_size:
                    problems_found.append(f"批次中样本数量不一致: {key}")
                    print(f"❌ 问题: 批次中样本数量不一致: {key}")
            
            # 6. 打印样本信息
            print("\n样本示例:")
            for i in range(min(2, batch_size)):  # 打印前2个样本
                sample = {k: v[i] for k, v in batch.items()}
                
                # 获取实际文本（如果有tokenizer）
                text = f"input_ids: {sample['input_ids'][:10]}..."  # 只显示前10个token
                try:
                    text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                except:
                    pass
                
                print(f"样本 {i}:")
                print(f"  文本: {text[:100]}...")  # 截断长文本
                print(f"  attention_mask: {sample['attention_mask'][:10]}...")
                print(f"  target: {sample['target']}")
                print(f"  prompt: {sample['prompts'][i][:50]}...")  # 显示部分prompt
                
        # 最终报告
        if not problems_found:
            print(f"\n✅ {loader_name} 数据检查完成，未发现明显问题")
        else:
            print(f"\n⛔ {loader_name} 发现问题汇总:")
            for problem in problems_found:
                print(f"  - {problem}")
    
    # # 检查训练集
    # inspect_data_loader(train_loader, "训练集")
    # # 检查验证集
    # inspect_data_loader(val_loader, "验证集")
    # # 检查测试集
    # inspect_data_loader(test_loader, "测试集")