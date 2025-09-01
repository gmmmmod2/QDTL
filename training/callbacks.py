import torch

class Callback:
    def on_train_begin(self, **kwargs):
        pass
    
    def on_epoch_begin(self, epoch, **kwargs):
        pass
    
    def on_batch_begin(self, batch, **kwargs):
        pass
    
    def on_batch_end(self, batch, **kwargs):
        pass
    
    def on_epoch_end(self, epoch, logs=None, **kwargs):
        pass
    
    def on_train_end(self, **kwargs):
        pass

class EarlyStoppingCallback(Callback):
    def __init__(self, patience=10, min_delta=0.001, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def on_epoch_end(self, epoch, logs=None, **kwargs):
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.best_score is None:
            self.best_score = current
        elif current < self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current
            self.counter = 0

class ModelCheckpointCallback(Callback):
    def __init__(self, filepath, logger, monitor='mean_Acc', save_best_only=True):
        self.filepath = filepath
        self.logger = logger
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_score = None
    
    def on_epoch_end(self, epoch, logs=None, **kwargs):
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.best_score is None or current > self.best_score:
            self.best_score = current
            torch.save({
                'epoch': epoch,
                'model_state_dict': kwargs['model'].state_dict(),
                'optimizer_state_dict': kwargs['optimizer'].state_dict(),
                'loss': current,
            }, self.filepath)
            self.logger.info("Best Result, save model ♪(´▽｀)")

class LRSchedulerCallback(Callback):
    def __init__(self, scheduler, monitor='val_loss'):
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_epoch_end(self, epoch, logs=None, **kwargs):
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(current)
        else:
            self.scheduler.step()

class LoggerCallback(Callback):
    def __init__(self, logger):
        self.logger = logger
    
    def on_train_begin(self, **kwargs):
        self.logger.info("Training started")
    
    def on_epoch_begin(self, epoch, **kwargs):
        self.logger.log_epoch_start(epoch, kwargs['total_epochs'])
    
    def on_epoch_end(self, epoch, logs=None, **kwargs):
        if logs:
            self.logger.log_metrics(logs, stage='val', epoch=epoch)
            self.logger.log_epoch_end(epoch, kwargs['total_epochs'])
    
    def on_train_end(self, **kwargs):
        self.logger.info("Training completed")

class CallbackHandler:
    def __init__(self, model, optimizer, config, logger, device, monitor='mean_Acc'):
        self.callbacks = []
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        self.device = device
        
        # 添加默认回调
        self.add_callback(LoggerCallback(logger))
        
        # 添加早停回调
        if 'early_stopping' in config['experiment']['training']:
            self.add_callback(EarlyStoppingCallback(
                patience=config['experiment']['training']['early_stopping']['patience'],
                min_delta=config['experiment']['training']['early_stopping']['min_delta'],
                monitor=monitor
            ))
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def on_train_begin(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)
    
    def on_epoch_begin(self, epoch, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)
    
    def on_batch_begin(self, batch, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, **kwargs)
    
    def on_batch_end(self, batch, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(batch, **kwargs)
    
    def on_epoch_end(self, epoch, logs=None, **kwargs):
        kwargs.update({
            'model': self.model,
            'config': self.config,
            'device': self.device,
            'optimizer': self.optimizer
        })
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs=logs, **kwargs)
    
    def on_train_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)
    
    def should_stop(self):
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback) and callback.early_stop:
                return True
        return False