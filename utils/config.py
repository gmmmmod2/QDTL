import json
import logging
from pathlib import Path
from typing import Any, Dict, Union
import yaml
import tomli

logger = logging.getLogger("utils.config")

class ConfigLoader:
    """Loads and merges multiple configuration files
    
    Supports JSON, YAML/YML, and TOML file formats. Handles:
    - File existence validation
    - Format detection by extension
    - Deep merging of configurations
    - Output file directory creation
    
    Attributes:
        merged_config (dict): Final merged configuration dictionary
        output_path (Path): Output path for merged configuration
    """
    def __init__(
        self, 
        *,
        model_config: Union[str, Path],
        experiment_config: Union[str, Path],
        datapath_config: Union[str, Path],
        output: Union[str, Path, None]):
        """Initialize the configuration loader
        
        Args:
            model_config: Path to model configuration file
            experiment_config: Path to experiment configuration file
            datapath_config: Path to data path configuration file
            output: Output path for merged configuration
        """
        # Load and validate all configurations
        config_paths = {
            "model": model_config,
            "experiment": experiment_config,
            "data": datapath_config
        }
        self._configs = {
            name: self._load_config(path)
            for name, path in config_paths.items()
        }
        
        # Merge configurations and save result
        self.merged_config = self._merge_configs()
        
        if output is None: 
            self.output_path = None
        else:
            self.output_path = Path(output).resolve()
            self.save_merged_config()
    
    def get_config(self) -> Dict[str, Any]:
        """Retrieve the merged configuration
        
        Returns:
            Current merged configuration dictionary
        """
        return self.merged_config
         
    def _load_config(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration file based on its extension
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Parsed configuration dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: For unsupported file formats
            RuntimeError: For parsing/reading errors
        """
        path = Path(file_path).resolve()
        self._validate_file(path)
        
        try:
            if path.suffix == '.json':
                return self._load_json(path)
            elif path.suffix in ['.yaml', '.yml']:
                return self._load_yaml(path)
            elif path.suffix == '.toml':
                return self._load_toml(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        except Exception as e:
            logger.exception(f"Failed to parse configuration: {path}")
            raise RuntimeError(f"Configuration parsing failed: {path}") from e

    def _merge_configs(self) -> Dict[str, Any]:
        """Merge all configuration dictionaries
        
        Returns:
            Unified configuration dictionary with namespaced sections
        """
        merged = {
            section: config 
            for section, config in self._configs.items()
        }
        logger.info("Configuration merging completed")
        return merged

    def save_merged_config(self):
        """Save merged configuration to output path as JSON
        
        Raises:
            OSError: For file writing issues
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.merged_config, f, indent=4, ensure_ascii=False)
            logger.info(f"Merged configuration saved to: {self.output_path}")
        except (OSError, TypeError) as e:
            logger.exception(f"Failed to save configuration: {self.output_path}")
            raise OSError(f"Configuration save failed: {self.output_path}") from e
    
    def _validate_file(self, path: Path):
        """Validate configuration file existence and type"""
        if not path.exists():
            logger.error(f"Config file not found: {path}")
            raise FileNotFoundError(f"Config file not found: {path}")
        if not path.is_file():
            logger.error(f"Config path is not a file: {path}")
            raise ValueError(f"Config path is not a file: {path}")
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration file"""
        with open(path, 'r', encoding='utf-8') as f:
            logger.debug(f"Loading JSON config: {path}")
            return json.load(f)
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(path, 'r', encoding='utf-8') as f:
            logger.debug(f"Loading YAML config: {path}")
            return yaml.safe_load(f)

    def _load_toml(self, path: Path) -> Dict[str, Any]:
        """Load TOML configuration file"""
        with open(path, 'rb') as f:
            logger.debug(f"Loading TOML config: {path}")
            return tomli.load(f)