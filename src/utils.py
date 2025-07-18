"""
Utility functions for resource monitoring and configuration loading.
"""

import psutil
import GPUtil
import logging
import yaml

def monitor_system_resources() -> None:
    """Monitor and log system resources"""
    logger = logging.getLogger(__name__)
    gpu_info = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if len(gpus) > 0:
                gpu = gpus[0]
                gpu_usage = gpu.memoryUsed / gpu.memoryTotal * 100
                gpu_info = f"GPU VRAM: {gpu_usage:.1f}% ({gpu.memoryUsed}MB/{gpu.memoryTotal}MB)"
            else:
                gpu_info = "GPU: No GPU detected"
        else:
            gpu_info = "GPU: CUDA not available"
    except Exception as e:
        gpu_info = f"GPU monitoring error: {str(e)}"
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"{gpu_info}, CPU: {cpu_percent:.1f}%, RAM: {memory.percent:.1f}%")

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
