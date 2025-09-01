from pathlib import Path
from datetime import datetime
from transformers import BertModel, BertTokenizer
import torch
import os

from utils.logger import TrainingLogger
from utils.other import get_model
from utils.config import ConfigLoader
from training.trainer import Trainer
from training.callbacks import LRSchedulerCallback, ModelCheckpointCallback
from training.optimization import create_optimizer, create_scheduler, create_loss_fn
from data.DataLoader import DataLoaderModule


PROJECT_ROOT = Path(__file__).resolve().parents[0]

def train(model_name, model_config, experiment_config, datapath_config, test_name: None):
    # ==> part one
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if test_name is not None:
        experiment_dir = PROJECT_ROOT / "experiments" / f"{test_name}"
    else:
        experiment_dir = PROJECT_ROOT / "experiments" / f"exp_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = experiment_dir / "training.log"
    config_file = experiment_dir / "config.json"
    checkpoint_file = experiment_dir / "best_model.pth"
    grad_file = experiment_dir / "grad.csv"

    logger = TrainingLogger(log_file)
    config_loader = ConfigLoader(
        model_config=model_config,
        experiment_config=experiment_config,
        datapath_config=datapath_config,
        output=config_file
    )
    config = config_loader.get_config()
    logger.log_system_info()
    logger.log_config(config)
    
    # ==> part two: 初始化训练组件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(config['model']['tokenizer'])
    
    # 创建数据加载器
    data_config = {
        'csv_path': config['data']['Datasets 1']['path'],
        'max_length': config['model']['max_length'],
        'batch_size': config['experiment']['training']['batch_size'],
        'train_ratio': config['data']['Datasets 1']['train_ratio'],
        'val_ratio': config['data']['Datasets 1']['val_ratio'],
        'test_ratio': config['data']['Datasets 1']['test_ratio'],
        'random_seed': config['experiment']['experiment']['seed'],
        'num_workers': min(4, os.cpu_count() - 1),
    }
    data_module = DataLoaderModule(data_config, tokenizer)
    train_loader, val_loader, _ = data_module.get_dataloaders()
    
    # 创建模型
    # ==> 冻结 bert 不跟随模型微调
    bert = BertModel.from_pretrained(config['model']['bert_model']).to(device)
    for param in bert.parameters():
        param.requires_grad = False
    bert.eval()  
    
    model = get_model(model_name, bert, tokenizer, device, config['model'])
    
    # 创建优化器和损失函数
    optimizer = create_optimizer(model, config['experiment']['optimizer'])
    scheduler, monitor_metric = create_scheduler(optimizer, config['experiment']['lr_scheduler'])
    loss_fn = create_loss_fn(config['experiment']['loss']['classification'])
    
    # 创建 Trainer
    monitor = 'mean_Acc'
    trainer = Trainer(model, bert, tokenizer, optimizer, device, config, logger, monitor, grad_file, is_grad=config['experiment']['experiment']['is_grad'])
     
    # 添加模型检查点回调
    trainer.callbacks.add_callback(ModelCheckpointCallback(
        filepath=checkpoint_file,
        monitor=monitor,
        logger=logger,
        save_best_only=True
    ))
    
    # 添加学习率调度器回调
    if scheduler:
        trainer.callbacks.add_callback(LRSchedulerCallback(
            scheduler=scheduler,
            monitor=monitor_metric
        ))
    logger.log_message("模型加载完成, 开始训练")
    
    # 训练循环
    total_epochs = config['experiment']['training']['epochs']
    trainer.callbacks.on_train_begin(total_epochs=total_epochs)
    
    for epoch in range(total_epochs):
        trainer.callbacks.on_epoch_begin(epoch, total_epochs=total_epochs)
        
        # 训练阶段
        train_loss = trainer.train_epoch(train_loader, optimizer, loss_fn)
        logger.log_metrics({'train_loss': train_loss}, stage='train', epoch=epoch)
        
        # 验证阶段
        val_loss, val_metrics = trainer.validate(val_loader, loss_fn)
        
        logs = {'val_loss': val_loss, **val_metrics}
        trainer.callbacks.on_epoch_end(epoch, logs=logs, total_epochs=total_epochs)
        
        # 检查是否早停
        if trainer.callbacks.should_stop():
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
        
    trainer.callbacks.on_train_end()
    logger.info(f"Training completed. Best model saved to {checkpoint_file}")

if __name__ == "__main__":
    model_name = 'QTRL'
    model_config = f"configs/{model_name}.toml"
    experiment_config = "configs/experiment.yaml"
    datapath_config = "configs/paths.json"
    train(model_name, model_config, experiment_config, datapath_config)