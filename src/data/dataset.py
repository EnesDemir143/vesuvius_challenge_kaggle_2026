import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


class VesuviusLMDBDataset(Dataset):
    def __init__(
        self,
        lmdb_path: Union[str, Path],
        transform: Optional[Callable] = None,
        split: str = "all",
        val_ratio: float = 0.2,
        seed: int = 42,
        return_metadata: bool = True
    ):
        self.lmdb_path = str(lmdb_path)
        self.transform = transform
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.return_metadata = return_metadata
        
        # Open LMDB in read-only mode
        self.env = None
        self._init_db()
        
        # Load volume keys
        with self.env.begin(write=False) as txn:
            keys_data = txn.get(b"__keys__")
            if keys_data is None:
                raise ValueError(f"Invalid LMDB database: __keys__ not found in {lmdb_path}")
            self.all_keys: List[str] = json.loads(keys_data.decode())
        
        # Apply train/val split
        self.keys = self._get_split_keys()
    
    def _init_db(self):
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
    
    def _get_split_keys(self) -> List[str]:
        """Get keys for the specified split using scroll-based stratification."""
        if self.split == "all":
            return self.all_keys
        
        # Build scroll_id -> volume_ids mapping
        scroll_to_volumes: Dict[int, List[str]] = {}
        with self.env.begin(write=False) as txn:
            for vol_id in self.all_keys:
                sid = int(txn.get(f"{vol_id}_scroll_id".encode()).decode())
                if sid not in scroll_to_volumes:
                    scroll_to_volumes[sid] = []
                scroll_to_volumes[sid].append(vol_id)
        
        # Sort scrolls by size (descending) for greedy balanced split
        sorted_scrolls = sorted(
            scroll_to_volumes.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        total_volumes = len(self.all_keys)
        target_val_size = int(total_volumes * self.val_ratio)
        
        # Greedy assignment: assign scrolls to val until we reach target
        rng = np.random.RandomState(self.seed)
        scroll_ids = [s[0] for s in sorted_scrolls]
        rng.shuffle(scroll_ids)  # Shuffle for randomness with seed
        
        val_scrolls = set()
        val_count = 0
        
        for sid in scroll_ids:
            scroll_size = len(scroll_to_volumes[sid])
            # Add to val if it doesn't exceed target too much
            if val_count + scroll_size <= target_val_size * 1.2:  # Allow 20% tolerance
                val_scrolls.add(sid)
                val_count += scroll_size
                if val_count >= target_val_size * 0.8:  # Stop if we reach 80% of target
                    break
        
        # Collect keys based on split
        train_keys = []
        val_keys = []
        
        for sid, vols in scroll_to_volumes.items():
            if sid in val_scrolls:
                val_keys.extend(vols)
            else:
                train_keys.extend(vols)
        
        if self.split == "val":
            return sorted(val_keys)
        elif self.split == "train":
            return sorted(train_keys)
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'train', 'val', or 'all'.")
    
    def __len__(self) -> int:
        return len(self.keys)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int, bool]]:
        vol_id = self.keys[idx]
        
        # Re-open environment if needed (for multiprocessing)
        if self.env is None:
            self._init_db()
        
        with self.env.begin(write=False) as txn:
            # Load image
            image_bytes = txn.get(f"{vol_id}_image".encode())
            image_shape = np.frombuffer(
                txn.get(f"{vol_id}_image_shape".encode()), dtype=np.int64
            )
            image_dtype = txn.get(f"{vol_id}_image_dtype".encode()).decode()
            
            image = np.frombuffer(image_bytes, dtype=image_dtype).reshape(image_shape)
            
            # Check if label exists
            has_label = txn.get(f"{vol_id}_has_label".encode()).decode() == "True"
            
            label = None
            if has_label:
                label_bytes = txn.get(f"{vol_id}_label".encode())
                label_shape = np.frombuffer(
                    txn.get(f"{vol_id}_label_shape".encode()), dtype=np.int64
                )
                label_dtype = txn.get(f"{vol_id}_label_dtype".encode()).decode()
                
                label = np.frombuffer(label_bytes, dtype=label_dtype).reshape(label_shape)
            
            # Load metadata
            scroll_id = int(txn.get(f"{vol_id}_scroll_id".encode()).decode())
        
        # Convert to tensors
        image = torch.from_numpy(image.copy()).float()
        if label is not None:
            label = torch.from_numpy(label.copy()).long()
        
        # Apply transforms
        if self.transform is not None:
            if label is not None:
                image, label = self.transform(image, label)
            else:
                image = self.transform(image)
        
        # Build output
        output = {
            'image': image,
            'label': label if label is not None else torch.zeros_like(image, dtype=torch.long),
        }
        
        if self.return_metadata:
            output['id'] = vol_id
            output['scroll_id'] = scroll_id
            output['has_label'] = has_label
        
        return output
    
    def __getstate__(self):
        """Handle pickling for multiprocessing DataLoader."""
        state = self.__dict__.copy()
        state['env'] = None  # Don't pickle LMDB environment
        return state
    
    def __setstate__(self, state):
        """Handle unpickling for multiprocessing DataLoader."""
        self.__dict__.update(state)
        self._init_db()
    
    def get_scroll_ids(self) -> List[int]:
        """Get unique scroll IDs in this dataset split."""
        scroll_ids = set()
        with self.env.begin(write=False) as txn:
            for vol_id in self.keys:
                sid = int(txn.get(f"{vol_id}_scroll_id".encode()).decode())
                scroll_ids.add(sid)
        return sorted(scroll_ids)
    
    def get_volumes_by_scroll(self, scroll_id: int) -> List[str]:
        """Get all volume IDs for a specific scroll."""
        volumes = []
        with self.env.begin(write=False) as txn:
            for vol_id in self.keys:
                sid = int(txn.get(f"{vol_id}_scroll_id".encode()).decode())
                if sid == scroll_id:
                    volumes.append(vol_id)
        return volumes
