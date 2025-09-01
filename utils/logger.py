import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

class TrainingLogger:
    """
    A flexible logging system for training processes that supports:
    - Configurable file rotation and console output
    - Structured logging of configurations, metrics, and hyperparameters
    - Error handling with stack trace capture
    - Custom formatting and log levels
    
    Example usage:
        from utils.logger import TrainingLogger
        
        logger = TrainingLogger(log_file="logs/training.log")
        logger.info("Starting training...")
        
        # Later when config is available
        config = load_config()
        logger.log_config(config)
    """
    
    def __init__(
        self,
        log_file: Union[str, Path] = "logs/training.log",
        log_level: Union[str, int] = logging.INFO,
        max_bytes: int = 20 * 1024 * 1024,  # 20 MB
        backup_count: int = 10,
        console_output: bool = True,
        log_format: Optional[str] = None,
        date_format: Optional[str] = None
    ):
        """
        Initialize the training logger.
        
        Args:
            log_file: Path to log file
            log_level: Logging level (string or int)
            max_bytes: Max log file size in bytes
            backup_count: Number of backup files to keep
            console_output: Enable console output
            log_format: Custom log format string
            date_format: Custom date format string
        """
        # Convert string log level to int
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Convert to Path and resolve
        self.log_path = Path(log_file).expanduser().resolve()
        self.log_dir = self.log_path.parent

        # Create log directory
        self._create_log_directory()
        
        # Configure logger
        self._configure_logger(
            log_level=log_level,
            max_bytes=max_bytes,
            backup_count=backup_count,
            console_output=console_output,
            log_format=log_format,
            date_format=date_format
        )
        
        # Get module-specific logger
        self.logger = logging.getLogger("utils.logger")
    
    def _create_log_directory(self):
        """Create log directory with error handling."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied creating log directory: {self.log_dir}"
            ) from e
        except OSError as e:
            raise OSError(
                f"Error creating log directory {self.log_dir}: {str(e)}"
            ) from e
        
        # Validate file location
        if self.log_path.is_dir():
            raise IsADirectoryError(
                f"Log path is a directory: {self.log_path}"
            )
    
    def _configure_logger(
        self,
        log_level: int,
        max_bytes: int,
        backup_count: int,
        console_output: bool,
        log_format: Optional[str],
        date_format: Optional[str]
    ):
        """Configure the logging system."""
        # Get root logger and set level
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
            finally:
                logger.removeHandler(handler)
        
        # Configure formatter
        default_log_format = (
            "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
        )
        default_date_format = "%Y-%m-%d %H:%M:%S"
        
        formatter = logging.Formatter(
            fmt=log_format or default_log_format,
            datefmt=date_format or default_date_format
        )
        
        # Setup file handler
        try:
            file_handler = RotatingFileHandler(
                filename=str(self.log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
                delay=True
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied for log file: {self.log_path}"
            ) from e
        except OSError as e:
            raise OSError(
                f"Error opening log file {self.log_path}: {str(e)}"
            ) from e
        
        # Add console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    def log_config(self, config: Dict[str, Any], title: str = "CONFIGURATION"):
        """
        Log a configuration dictionary in a structured format.
        
        Args:
            config: Configuration dictionary to log
            title: Optional title for the configuration section
            
        Example:
            logger.log_config({
                "training": {
                    "epochs": 100,
                    "batch_size": 32
                },
                "model": {
                    "name": "resnet50"
                }
            })
        """
        self.logger.info(f"===== {title} =====")
        self._log_nested_dict(config)
        self.logger.info("=" * (len(title) + 12))
    
    def _log_nested_dict(self, data: Dict[str, Any], indent: int = 0):
        """Recursively log nested dictionary structures."""
        indent_str = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                self.logger.info(f"{indent_str}{key}:")
                self._log_nested_dict(value, indent + 1)
            elif isinstance(value, list):
                self.logger.info(f"{indent_str}{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self.logger.info(f"{indent_str}  [{i}]:")
                        self._log_nested_dict(item, indent + 2)
                    else:
                        self.logger.info(f"{indent_str}  - {item}")
            else:
                self.logger.info(f"{indent_str}{key}: {value}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters in a structured format.
        
        Args:
            hyperparams: Dictionary of hyperparameter names and values
        """
        self.logger.info("===== HYPERPARAMETERS =====")
        for name, value in hyperparams.items():
            self.logger.info(f"  {name}: {value}")
        self.logger.info("=" * 25)
    
    def log_metrics(self, metrics: Dict[str, float], stage: str = "", epoch: Optional[int] = None):
        """
        Log training/evaluation metrics in a structured format.
        
        Args:
            metrics: Dictionary of metric names and values
            stage: Training stage (e.g., 'train', 'val', 'test')
            epoch: Current epoch number (if applicable)
        """
        # Build title
        title_parts = []
        if epoch is not None:
            title_parts.append(f"EPOCH {epoch}")
        if stage:
            title_parts.append(stage.upper())
        title_parts.append("METRICS")
        title = " | ".join(title_parts)
        
        self.logger.info(f"===== {title} =====")
        for name, value in metrics.items():
            # Format floats nicely, leave others as-is
            if isinstance(value, float):
                self.logger.info(f"  {name}: {value:.6f}")
            else:
                self.logger.info(f"  {name}: {value}")
        self.logger.info("=" * (len(title) + 12))
    
    def log_table(self, headers: List[str], rows: List[List[Any]], title: str = "TABLE"):
        """
        Log tabular data in a formatted table.
        
        Args:
            headers: List of column headers
            rows: List of rows (each row is a list of values)
            title: Optional title for the table
        """
        # Convert all values to strings
        str_rows = [[str(item) for item in row] for row in rows]
        str_headers = [str(h) for h in headers]
        
        # Calculate column widths
        col_widths = [len(h) for h in str_headers]
        for row in str_rows:
            for i, item in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(item))
        
        # Build header row
        header_row = " | ".join(
            h.ljust(w) for h, w in zip(str_headers, col_widths)
        )
        
        # Build separator
        separator = "-+-".join("-" * w for w in col_widths)
        
        # Log the table
        self.logger.info(f"===== {title} =====")
        self.logger.info(header_row)
        self.logger.info(separator)
        
        for row in str_rows:
            row_str = " | ".join(
                item.ljust(w) for item, w in zip(row, col_widths[:len(row)])
            )
            self.logger.info(row_str)
        
        self.logger.info("=" * (len(title) + 12))
    
    def log_message(self, message: str, level: str = "info", border: bool = True):
        """
        Log a formatted message with optional border.
        
        Args:
            message: Message text to log
            level: Log level ('debug', 'info', 'warning', 'error', 'critical')
            border: Add decorative border around the message
        """
        log_func = getattr(self.logger, level, self.logger.info)
        
        if border:
            border_line = "=" * 80
            log_func(border_line)
            log_func(f"{message.center(80)}")
            log_func(border_line)
        else:
            log_func(message)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log a standardized message for epoch start."""
        self.logger.info(f"════════════════════ EPOCH {epoch}/{total_epochs} ════════════════════")
    
    def log_epoch_end(self, epoch: int, total_epochs: int):
        """Log a standardized message for epoch end."""
        self.logger.info(f"═════════════════ COMPLETED EPOCH {epoch}/{total_epochs} ═════════════════")
    
    def log_experiment_info(self, experiment_name: str, run_id: str):
        """Log experiment identification information."""
        self.logger.info(f"EXPERIMENT: {experiment_name}")
        self.logger.info(f"RUN ID: {run_id}")
        self.logger.info("=" * 50)
    
    def log_system_info(self):
        """Log system and environment information."""
        import sys
        import platform
        import torch
        
        self.logger.info("===== SYSTEM INFORMATION =====")
        self.logger.info(f"Python: {sys.version.split()[0]}")
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info(f"PyTorch: {torch.__version__}")
        self.logger.info(f"CUDA: {torch.version.cuda}") if torch.cuda.is_available() else None
        self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}") if torch.cuda.is_available() else None
        self.logger.info(f"CPU cores: {os.cpu_count()}")
        self.logger.info(f"Working dir: {Path.cwd()}")
        self.logger.info("=" * 25)
    
    # Delegate common logging methods to the actual logger
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)