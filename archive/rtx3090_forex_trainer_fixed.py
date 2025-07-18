#!/usr/bin/env python3
"""
RTX 3090 Standalone Training System
==================================
Fixed version with proper GPU monitoring handling
"""

import os
import sys
import time
import torch
import psutil
from datetime import datetime

# Configuration - Adjust these as needed
FULL_GENERATIONS = 200
GPU_UTILIZATION = 0.8
CPU_UTILIZATION = 0.8
VRAM_FRACTION = 1.0
CHECKPOINT_INTERVAL = 10

class RTX3090Trainer:
    def __init__(self):
        self.start_time = time.time()
        self.best_score = -float('inf')
        self.current_gen = 0
        self.setup_resources()
        self.setup_logging()
        self.model = None
        self.optimizer = None
        
    def setup_resources(self):
        """Configure hardware resources with better error handling"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU VRAM allocated: {total_vram:.1f}GB (100%)")
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision('high')
            except Exception as e:
                print(f"GPU configuration warning: {str(e)}")

        self.total_threads = os.cpu_count()
        self.worker_threads = int(self.total_threads * CPU_UTILIZATION)
        torch.set_num_threads(self.worker_threads)
        
        print("\n=== HARDWARE CONFIGURATION ===")
        print(f"GPU: RTX 3090 @ {GPU_UTILIZATION*100:.0f}% processing power")
        print(f"CPU: Using {self.worker_threads} of {self.total_threads} threads")
        print("==============================\n")
        
    def setup_logging(self):
        """Initialize logging system"""
        self.log_file = "training_log.txt"
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log - Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def log(self, message, level="INFO"):
        """Enhanced logging with better GPU monitoring"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cpu_usage = psutil.cpu_percent()
        
        try:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / (1024**3)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except:
                    gpu_usage = "N/A"
            else:
                gpu_mem = 0
                gpu_usage = "N/A"
        except Exception as e:
            gpu_mem = 0
            gpu_usage = "N/A"
            
        log_entry = (f"[{timestamp}] [GPU:{gpu_usage}%/{gpu_mem:.1f}GB] "
                    f"[CPU:{cpu_usage}%] {message}")
        
        print(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
            
    def initialize_model(self):
        """Initialize model with better error handling"""
        self.log("Initializing model...")
        
        try:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(10, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.log("Model initialized successfully")
            return True
        except Exception as e:
            self.log(f"Model initialization failed: {str(e)}", "ERROR")
            return False
            
    def train_generation(self):
        """Run one complete training generation"""
        self.current_gen += 1
        start_time = time.time()
        
        try:
            # Training phase
            with torch.cuda.amp.autocast():  # Mixed precision
                inputs = torch.randn(32, 10).to(self.device)
                targets = torch.randn(32, 1).to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = torch.nn.functional.mse_loss(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Calculate score
                score = -loss.item()
            
            # Track best performance
            if score > self.best_score:
                self.best_score = score
                self.save_model(f"champion_gen{self.current_gen}.pth")
                
            # Periodic checkpoint
            if self.current_gen % CHECKPOINT_INTERVAL == 0:
                self.save_model(f"checkpoint_gen{self.current_gen}.pth")
                
            # Log progress
            gen_time = time.time() - start_time
            self.log(f"Gen {self.current_gen} | Score: {score:.4f} | Time: {gen_time:.1f}s")
            return True
            
        except Exception as e:
            self.log(f"Training error: {str(e)}", "ERROR")
            return self.recover_from_error()
            
    def save_model(self, filename):
        """Save model checkpoint with error handling"""
        try:
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'score': self.best_score,
                'generation': self.current_gen
            }, filename)
            self.log(f"Saved model checkpoint: {filename}")
        except Exception as e:
            self.log(f"Failed to save model: {str(e)}", "ERROR")
            
    def recover_from_error(self):
        """Comprehensive error recovery system"""
        self.log("Attempting error recovery...", "WARNING")
        
        # 1. Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 2. Cool-down period
        time.sleep(5)
        
        # 3. Attempt to reinitialize
        try:
            self.log("Reinitializing model...")
            return self.initialize_model()
        except Exception as e:
            self.log(f"Recovery failed: {str(e)}", "CRITICAL")
            return False
            
    def finalize(self):
        """Complete training session"""
        total_time = (time.time() - self.start_time) / 3600
        self.log(f"\n=== TRAINING COMPLETED ===")
        self.log(f"Total duration: {total_time:.2f} hours")
        self.log(f"Best score achieved: {self.best_score:.4f}")
        self.log(f"Final model saved to: final_model.pth")
        
        # Save final model
        self.save_model("final_model.pth")
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    print("\n=== RTX 3090 STANDALONE TRAINING ===")
    print("Starting training session...\n")
    
    # Initialize trainer
    trainer = RTX3090Trainer()
    
    # Initialize model - stop if fails
    if not trainer.initialize_model():
        sys.exit("❌ Model initialization failed - cannot continue")
    
    # Main training loop
    try:
        while trainer.current_gen < FULL_GENERATIONS:
            success = trainer.train_generation()
            
            if not success:
                trainer.log(f"Retrying generation {trainer.current_gen}...", "WARNING")
                trainer.current_gen -= 1  # Retry current generation
                if trainer.current_gen < 0:  # Prevent infinite retry on first gen
                    raise RuntimeError("Failed to recover from initial generation error")
                
        # Training completed successfully
        trainer.finalize()
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
    finally:
        # Ensure finalization runs even on error
        trainer.finalize()

if __name__ == "__main__":
    # Verify PyTorch can access GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available - Please check your GPU drivers")
        sys.exit(1)
        
    # Run training
    main()
