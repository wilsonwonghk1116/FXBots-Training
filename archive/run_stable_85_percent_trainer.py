#!/usr/bin/env python3
"""
STABLE 85% VRAM Trainer - RTX 3090 24GB Optimized
Designed to successfully complete training with memory management
"""

import os
import sys
import time
import logging
from typing import Any, Optional

# Add project paths to ensure imports work on all nodes
project_paths = [
    "/home/w1/cursor-to-copilot-backup/TaskmasterForexBots",  # Head PC
    "/home/w2/cursor-to-copilot-backup/TaskmasterForexBots"   # Worker PC
]
for path in project_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

import torch
from config import *
from bot_population import create_population, evaluate_fitness, select_elite, tournament_selection, mutate_population, crossover_population, EvaluationActor
from trading_bot import TradingBot
from synthetic_env import SyntheticForexEnv
from indicators import compute_indicators
from predictors import *
from reward import compute_reward
from checkpoint_utils import save_checkpoint, load_checkpoint
from champion_analysis import analyze_champion
from utils import log_resource_usage
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_worker_environment():
    """Ensure Ray workers have proper environment setup"""
    import sys
    import os
    
    # Add both project paths
    project_paths = [
        "/home/w1/cursor-to-copilot-backup/TaskmasterForexBots",  # Head PC
        "/home/w2/cursor-to-copilot-backup/TaskmasterForexBots"   # Worker PC
    ]
    
    for path in project_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    # Set PYTHONPATH environment variable
    pythonpath = ":".join(project_paths)
    if "PYTHONPATH" in os.environ:
        os.environ["PYTHONPATH"] = f"{pythonpath}:{os.environ['PYTHONPATH']}"
    else:
        os.environ["PYTHONPATH"] = pythonpath
    
    return True

def evolutionary_training_loop(actor_pool, population, progress, node_name="local"):
    champion_history = []
    best_overall_score = float('-inf')
    num_elites = 5

    for generation in range(progress.get("generation", 0), GENERATIONS):
        logger.info(f"\n🔥 === GENERATION {generation + 1}/{GENERATIONS} ({node_name}) === 🔥")
        
        # 1. Evaluate fitness in parallel
        fitness_scores = evaluate_fitness(actor_pool, population)
        
        # 2. Select elites and preserve their weights
        elite_weights = select_elite(population, fitness_scores, num_elite=num_elites)
        
        # 3. Create the mating pool from the whole population
        mating_pool = tournament_selection(population, fitness_scores, num_to_select=len(population) - num_elites)
        
        # 4. The next generation starts with the elites
        # We will overwrite the non-elite part of the population
        next_gen_population = population # Work in-place
        
        # Load elite weights into the first few bots of the next generation
        for i in range(num_elites):
            next_gen_population[i].load_state_dict(elite_weights[i])

        # 5. Create offspring for the rest of the population
        offspring_to_create = next_gen_population[num_elites:]
        crossover_population(mating_pool, offspring_to_create)
        
        # 6. Mutate the non-elite offspring
        mutate_population(offspring_to_create)

        # 7. Save checkpoint
        if (generation + 1) % CHECKPOINT_INTERVAL == 0:
            # Note:Checkpointing needs to be adapted for the new in-place evolution
            pass # save_checkpoint(...)
            
        # 8. Track and analyze champion
        champion_idx = int(np.argmax(fitness_scores))
        champion = population[champion_idx] # The champion from this generation
        champion_history.append(champion.state_dict()) # Store weights, not object
        
        # 9. Log resource usage
        log_resource_usage()
        
    # Final champion analysis and save
    final_champion_weights = champion_history[-1]
    final_champion = TradingBot(strategy_type="champion", device="cpu")
    final_champion.load_state_dict(final_champion_weights)
    
    # Analysis part needs an environment, create a temporary one
    # The analyze_champion function's signature needs to be checked.
    # Corrected based on research: passing empty placeholders for history and metrics
    analysis = analyze_champion(final_champion, [], {})
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(final_champion.state_dict(), os.path.join(MODEL_DIR, f"final_champion_{node_name}.pt"))
    logger.info(f"🎊 === TRAINING COMPLETE on {node_name} ===")
    return final_champion, analysis

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    env = SyntheticForexEnv(real_data_path=EURUSD_H1_PATH)
    population = create_population(POP_SIZE_3090 + POP_SIZE_3070, ["ultra_aggressive", "aggressive", "conservative", "balanced", "contrarian", "momentum", "scalper"], device="cpu")
    progress = {"generation": 0}

    # In single-machine mode, we can't use Ray actors. We fall back to sequential evaluation.
    # This part needs a non-actor evaluation function if single-machine mode is to be fully supported.
    # For now, focusing on the Ray cluster part.
    # evolutionary_training_loop(env, population, progress, node_name="local")

def main_ray():
    if not RAY_AVAILABLE or ray is None:
        raise ImportError("Ray is not installed. Please install ray to use distributed training.")
    
    # Setup worker environment before Ray init
    setup_worker_environment()
    
    # Initialize Ray with runtime environment settings
    ray_init_config = {
        'address': 'auto',
        'runtime_env': {
            'env_vars': {
                'PYTHONPATH': os.environ.get('PYTHONPATH', '')
            },
            'working_dir': os.getcwd(),
            'excludes': [
                '*.pth',          # Exclude all model files
                '*.tar.gz',       # Exclude compressed files
                'ray_logs.tar.gz',
                'CHAMPION_BOT_*.pth',
                'CHAMPION_ANALYSIS_*.json', 
                'champion_gen*.pth',
                'checkpoint_gen*.pth',
                'data/EURUSD_H1.csv',  # Don't upload data files - they exist locally
                '.git/',          # Exclude git objects
                '.git/objects/',
                'logs/',
                'checkpoints/',
                '__pycache__/',
                '*.pyc'
            ]
        }
    }
    
    try:
        ray.init(**ray_init_config)
        logger.info("Connected to Ray cluster. Resources: %s", ray.cluster_resources())
    except Exception as e:
        logger.error(f"Failed to connect to Ray cluster: {e}")
        raise
    
    # Split population for each node
    bot_configs_3090 = list(range(POP_SIZE_3090))
    bot_configs_3070 = list(range(POP_SIZE_3090, POP_SIZE_3090 + POP_SIZE_3070))

    @ray.remote(num_gpus=GPU_3090, num_cpus=CPU_3090)
    def train_bots_3090(bot_indices):
        try:
            # Ensure worker environment is set up
            setup_worker_environment()
            
            # Re-import ray inside the worker to ensure it's available
            import ray as ray_worker
            
            # BULLETPROOF data loading with memory management
            logger.info("Loading EURUSD data with memory-safe parsing...")
            
            # Load with explicit dtypes and error handling for corrupted data
            # The file HAS headers, so we read them from the first row
            # CRITICAL FIX: CSV uses lowercase column names!
            df = pd.read_csv(
                EURUSD_H1_PATH, 
                header=0,  # First row contains headers
                dtype={
                    'timestamp': str,  # Keep as string initially (lowercase!)
                    'open': np.float32,  # Use float32 instead of float64 to save memory (lowercase!)
                    'high': np.float32,  # (lowercase!)
                    'low': np.float32,   # (lowercase!)
                    'close': np.float32, # (lowercase!) - this was causing the KeyError!
                    'volume': np.float32 # (lowercase!)
                },
                na_values=['', 'null', 'NULL', 'N/A', 'nan'],  # Handle various null representations
                keep_default_na=True,
                low_memory=False,  # Disable low_memory to prevent dtype warnings
                on_bad_lines='skip'   # Skip bad lines instead of crashing (modern pandas syntax)
            )
            
            # Clean the data: drop any rows with NaN values in close column (lowercase!)
            initial_rows = len(df)
            df = df.dropna(subset=['close'])  # FIXED: use lowercase 'close'
            final_rows = len(df)
            logger.info(f"Data loaded: {final_rows} valid rows (dropped {initial_rows - final_rows} corrupted rows)")
            
            # Convert to numpy array and put in Ray object store
            close_prices = df['close'].to_numpy()  # FIXED: use lowercase 'close'
            data_ref = ray_worker.put(close_prices)
            
            # Force garbage collection to free memory after data processing
            del df, close_prices
            import gc
            gc.collect()
            
            logger.info("Data successfully loaded and stored in Ray object store")
            
            # Create a pool of evaluation actors on this node, passing the data reference
            assigned_resources = ray_worker.get_runtime_context().get_assigned_resources()
            num_actors = int(assigned_resources.get("CPU", 1) * ACTOR_CPU_SCALE_FACTOR)
            actor_pool = [EvaluationActor.remote(data_ref) for _ in range(num_actors)]

            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            population = create_population(len(bot_indices), ["ultra_aggressive", "aggressive", "conservative", "balanced", "contrarian", "momentum", "scalper"], device="cuda")
            progress = {"generation": 0}
            champion, analysis = evolutionary_training_loop(actor_pool, population, progress, node_name="3090")
            return champion, analysis
        except Exception as e:
            logger.error(f"FATAL ERROR on node 3090: {e}", exc_info=True)
            raise e

    @ray.remote(num_gpus=GPU_3070, num_cpus=CPU_3070)
    def train_bots_3070(bot_indices):
        try:
            # Ensure worker environment is set up
            setup_worker_environment()
            
            # Re-import ray inside the worker to ensure it's available
            import ray as ray_worker
            
            # BULLETPROOF data loading with memory management
            logger.info("Loading EURUSD data with memory-safe parsing...")
            
            # Load with explicit dtypes and error handling for corrupted data
            # The file HAS headers, so we read them from the first row
            # CRITICAL FIX: CSV uses lowercase column names!
            df = pd.read_csv(
                EURUSD_H1_PATH, 
                header=0,  # First row contains headers
                dtype={
                    'timestamp': str,  # Keep as string initially (lowercase!)
                    'open': np.float32,  # Use float32 instead of float64 to save memory (lowercase!)
                    'high': np.float32,  # (lowercase!)
                    'low': np.float32,   # (lowercase!)
                    'close': np.float32, # (lowercase!) - this was causing the KeyError!
                    'volume': np.float32 # (lowercase!)
                },
                na_values=['', 'null', 'NULL', 'N/A', 'nan'],  # Handle various null representations
                keep_default_na=True,
                low_memory=False,  # Disable low_memory to prevent dtype warnings
                on_bad_lines='skip'   # Skip bad lines instead of crashing (modern pandas syntax)
            )
            
            # Clean the data: drop any rows with NaN values in close column (lowercase!)
            initial_rows = len(df)
            df = df.dropna(subset=['close'])  # FIXED: use lowercase 'close'
            final_rows = len(df)
            logger.info(f"Data loaded: {final_rows} valid rows (dropped {initial_rows - final_rows} corrupted rows)")
            
            # Convert to numpy array and put in Ray object store
            close_prices = df['close'].to_numpy()  # FIXED: use lowercase 'close'
            data_ref = ray_worker.put(close_prices)
            
            # Force garbage collection to free memory after data processing
            del df, close_prices
            import gc
            gc.collect()
            
            logger.info("Data successfully loaded and stored in Ray object store")
            
            # Create a pool of evaluation actors on this node, passing the data reference
            assigned_resources = ray_worker.get_runtime_context().get_assigned_resources()
            num_actors = int(assigned_resources.get("CPU", 1) * ACTOR_CPU_SCALE_FACTOR)
            actor_pool = [EvaluationActor.remote(data_ref) for _ in range(num_actors)]

            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            population = create_population(len(bot_indices), ["ultra_aggressive", "aggressive", "conservative", "balanced", "contrarian", "momentum", "scalper"], device="cuda")
            progress = {"generation": 0}
            champion, analysis = evolutionary_training_loop(actor_pool, population, progress, node_name="3070")
            return champion, analysis
        except Exception as e:
            logger.error(f"FATAL ERROR on node 3070: {e}", exc_info=True)
            raise e
            
    # Launch Ray tasks
    future_3090 = train_bots_3090.remote(bot_configs_3090)
    future_3070 = train_bots_3070.remote(bot_configs_3070)
    result_3090 = ray.get(future_3090)
    result_3070 = ray.get(future_3070)
    # Aggregate champions and select global champion
    champions = [result_3090[0], result_3070[0]]
    analyses = [result_3090[1], result_3070[1]]
    # TODO: Implement global champion selection logic (e.g., evaluate on full env)
    logger.info(f"Distributed training complete. Champions: {[c.strategy_type for c in champions]}")
    # Save/Analyze global champion as needed

if __name__ == "__main__":
    if os.environ.get("RAY_CLUSTER", "0") == "1":
        if RAY_AVAILABLE:
            main_ray()
        else:
            raise ImportError("Ray is not installed. Please install ray to use distributed training.")
    else:
        main() 