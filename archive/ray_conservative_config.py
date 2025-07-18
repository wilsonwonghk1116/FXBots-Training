# Optimized Ray Cluster Configuration
# Conservative resource allocation to prevent conflicts

import ray

# Resource configuration for RTX 3070 + RTX 3090 setup
CONSERVATIVE_CONFIG = {
    "pc1_resources": {
        "num_cpus": 15,  # Reduced from 20
        "num_gpus": 0.8,  # Reduced from 1.0
        "memory": 12 * 1024 * 1024 * 1024,  # 12GB
        "object_store_memory": 4 * 1024 * 1024 * 1024  # 4GB
    },
    "pc2_resources": {
        "num_cpus": 12,  # Reduced from 20
        "num_gpus": 0.7,  # Reduced from 1.0  
        "memory": 8 * 1024 * 1024 * 1024,  # 8GB
        "object_store_memory": 3 * 1024 * 1024 * 1024  # 3GB
    }
}

# Actor configuration
ACTOR_CONFIG = {
    "cpu_per_actor": 15,  # Reduced from 20
    "gpu_per_actor": 0.8,  # Reduced from 1.0
    "max_actors": 2  # Ensure we only create 2 actors total
}

def get_conservative_resources():
    """Get conservative resource allocation"""
    return CONSERVATIVE_CONFIG

def get_actor_config():
    """Get actor configuration"""
    return ACTOR_CONFIG 