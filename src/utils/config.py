from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import yaml


@dataclass
class LabelEncoding:
    background: int = 0
    foreground: int = 1
    unlabeled: int = 2


@dataclass
class DataConfig:
    raw_data_dir: str = "dataset/raw"
    train_csv: str = "dataset/raw/train.csv"
    test_csv: str = "dataset/raw/test.csv"
    train_images: str = "dataset/raw/train_images"
    train_labels: str = "dataset/raw/train_labels"
    test_images: str = "dataset/raw/test_images"
    processed_dir: str = "dataset/processed"
    train_lmdb: str = "dataset/processed/train.lmdb"
    test_lmdb: str = "dataset/processed/test.lmdb"
    label_encoding: LabelEncoding = field(default_factory=LabelEncoding)
    val_ratio: float = 0.2
    seed: int = 42
    
    def get_path(self, key: str, root: Optional[Path] = None) -> Path:
        """Get absolute path for a config key."""
        value = getattr(self, key)
        path = Path(value)
        if root and not path.is_absolute():
            path = root / path
        return path


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    _root: Optional[Path] = field(default=None, repr=False)
    
    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "Config":
        """Load config from YAML file."""
        config_file = Path(config_path)
        
        # Find project root (where config.yaml is)
        if not config_file.is_absolute():
            # Search upward from cwd
            current = Path.cwd()
            while current != current.parent:
                if (current / config_path).exists():
                    config_file = current / config_path
                    break
                current = current.parent
        
        root = config_file.parent
        
        with open(config_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Parse data config
        data_dict = yaml_data.get('data', {})
        label_enc = data_dict.pop('label_encoding', {})
        label_encoding = LabelEncoding(**label_enc)
        data_config = DataConfig(**data_dict, label_encoding=label_encoding)
        
        return cls(data=data_config, _root=root)
    
    def path(self, key: str) -> Path:
        """Get absolute path for a data config key."""
        return self.data.get_path(key, self._root)


# Singleton instance - loaded once, used everywhere
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance (lazy loaded)."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def load_config(config_path: str = "config.yaml") -> Config:
    """Load config from specific path and set as global."""
    global _config
    _config = Config.load(config_path)
    return _config