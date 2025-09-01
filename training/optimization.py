"""
OPTIMIZATION UTILITIES MODULE

This module handles dynamic creation of training components:
- Optimizers
- Learning Rate Schedulers
- Loss Functions
- Gradient clipping

Design principles: 
1. Configuration-driven initialization
2. Type safety
3. Extensibility

Example usage: 
>>> from training.optimization import create_optimizer, create_scheduler, create_loss_fn
>>> optimizer = create_optimizer(model, config['optimizer'])
>>> scheduler = create_scheduler(optimizer, config['lr_scheduler'])
>>> loss_fn = create_loss_fn(config['loss'])
"""
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from typing import Any, Dict, Optional, Tuple, Type, Union


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Create optimizer from configuration
    
    Args:
        model: Model containing parameters to optimize
        config: Optimizer configuration
        
    Returns:
        Initialized optimizer
        
    Raises:
        ValueError: Invalid configuration or missing required parameters
    """
    # Validate configuration
    if not config or "type" not in config:
        raise ValueError("Optimizer configuration must contain 'type' key")
    
    optimizer_type = config["type"]
    params = config.get("params", {})
    
    # Get optimizer class
    try:
        optimizer_class: Type[optim.Optimizer] = getattr(optim, optimizer_type)
        return optimizer_class(model.parameters(), **params)
    except AttributeError as e:
        raise ValueError(f"Optimizer '{optimizer_type}' not found") from e
    except TypeError as e:
        raise ValueError(f"Invalid parameters for {optimizer_type}: {e}") from e

def create_scheduler(
    optimizer: optim.Optimizer, 
    config: Optional[Dict[str, Any]] = None
) -> Union[Tuple[_LRScheduler, str], _LRScheduler, None]:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer to attach to
        config: Scheduler configuration (None returns None)
        
    Returns:
        - ReduceLROnPlateau: (scheduler, monitor_metric)
        - Other schedulers: scheduler instance
        - None if no configuration
        
    Raises:
        ValueError: Invalid configuration
    """
    if not config:
        return None
    
    # Validate configuration
    if "type" not in config:
        raise ValueError("Scheduler configuration must contain 'type' key")
    
    scheduler_type = config["type"]
    params = config.get("params", {})
    
    # Handle ReduceLROnPlateau separately
    try:
        # Handle special schedulers
        if scheduler_type == "ReduceLROnPlateau":
            monitor = params.pop("monitor", "val_loss")
            return (ReduceLROnPlateau(optimizer, **params), monitor)
        
        # Handle standard schedulers
        scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
        return scheduler_class(optimizer, **params)
        
    except AttributeError as e:
        raise ValueError(f"Scheduler '{scheduler_type}' not found") from e
    except TypeError as e:
        raise ValueError(f"Invalid parameters for {scheduler_type}: {e}") from e

def create_loss_fn(config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Create loss function from configuration
    
    Args:
        config: Loss configuration
        
    Returns:
        Loss function instance
        
    Raises:
        ValueError: Invalid loss type
    """
    # Set default loss function
    if not config or config is None:
        raise ValueError("The configuration file must be passed in")
    elif "type" not in config:
        raise ValueError("Loss function configuration must contain 'type' key")
    
    loss_type = config["type"]
    params = config.get("params", {})
    
    try:
        loss_class = getattr(nn, loss_type)
        return loss_class(**params)
    except AttributeError as e:
        raise ValueError(f"Loss function '{loss_type}' not found") from e
    except TypeError as e:
        raise ValueError(f"Invalid parameters for {loss_type}: {e}") from e

def apply_gradient_clipping(model: nn.Module, clip_value: Union[int, float]) -> None:
    """
    Apply gradient clipping to the model parameters if clip_value is positive.
    
    This function uses the `clip_grad_norm_` utility from PyTorch to prevent
    exploding gradients during training. Clipping is only applied if the
    specified clip_value is greater than 0.
    
    Args:
        model: PyTorch model whose gradients will be clipped
        clip_value: Maximum norm of the gradients. If <= 0, no clipping is performed.
            Typical values range from 0.1 to 10.0, depending on the model architecture.
            
    Raises:
        TypeError: If clip_value is not a numeric type
        RuntimeError: If no gradients are computed (backward pass not performed)
        
    Example:
        >>> model = nn.Linear(10, 2)
        >>> optimizer = optim.SGD(model.parameters(), lr=0.01)
        >>> # Inside training loop:
        >>> loss = model(input)
        >>> loss.backward()
        >>> apply_gradient_clipping(model, clip_value=1.0)
        >>> optimizer.step()
        
    Note:
        - Should be called after `backward()` and before `optimizer.step()`
        - Only affects parameters with `requires_grad=True`
        - Clips gradients in-place using the L2 norm (Euclidean norm)
    """
    if not isinstance(clip_value, (int, float)):
        raise TypeError(f"clip_value must be numeric, got {type(clip_value)}")
    if clip_value > 0:
        nn.utils.clip_grad_norm_(model.parameters(), clip_value, norm_type=2.0)