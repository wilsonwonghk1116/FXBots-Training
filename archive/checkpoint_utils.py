"""
checkpoint_utils.py
Checkpointing utilities for saving/loading population and training state.
"""
import torch
import pickle

def save_checkpoint(path, population, env_state, progress):
    """Save population, environment state, and progress to disk."""
    torch.save({'population': population, 'env_state': env_state, 'progress': progress}, path)

def load_checkpoint(path):
    """Load checkpoint from disk."""
    return torch.load(path)
# Add extension points for more complex checkpointing. 