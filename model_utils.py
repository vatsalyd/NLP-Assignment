import pickle
import os
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

def save_model(obj, name: str):
    """Save model to checkpoint directory."""
    path = CHECKPOINT_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved checkpoint: {path}")
    return path

def load_model(name: str):
    """Load model from checkpoint directory."""
    path = CHECKPOINT_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Loaded checkpoint: {path}")
    return obj

def checkpoint_exists(name: str) -> bool:
    """Check if checkpoint exists."""
    path = CHECKPOINT_DIR / f"{name}.pkl"
    return path.exists()

def get_or_train(name: str, train_fn, *args, force_retrain: bool = False, **kwargs):
    """Load from checkpoint or train and save."""
    if not force_retrain and checkpoint_exists(name):
        return load_model(name)
    obj = train_fn(*args, **kwargs)
    save_model(obj, name)
    return obj