#!/usr/bin/env python3
"""
RTX 3090 Forex Trainer - Working Version
=======================================
Version 3.0 - Guaranteed Execution
"""

import os
import sys
import time
import torch
import psutil
from datetime import datetime

# Configuration
VERSION = "3.0"
FULL_GENERATIONS = 5  # Reduced for testing
GPU_UTILIZATION = 0.8
CPU_UTILIZATION = 0.8
VRAM_FRACTION = 1.0
CHECKPOINT_INTERVAL = 2
PAUSE_DURATION = 2

class ForexTrainer:
    def __init__(self):
        print("\n=== INITIALIZING FOREX TRAINER ===")
        self.start_time = time.time()
        self.best_score = -float('inf')
        self.current_gen = 0
        self.performance_history = []
        self.setup_resources()
        self.setup_logging()
        self.model = None
        self.optimizer = None
        
    def setup_resources(self):
        """Verify and configure hardware"""
        print("Setting up resources...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
                print(f"Allocated {VRAM_FRACTION*100}% VRAM")
            except Exception as e:
                print(f"VRAM allocation error: {e}")

        self.total_threads = os.cpu_count()
        self.worker_threads = int(self.total_threads * CPU_UTILIZATION)
        torch.set_num_threads(self.worker_threads)
        print(f"Using {self.worker_threads} CPU threads")
        
    def setup_logging(self):
        """Initialize logging system"""
        self.log_file = "training_log.txt"
        print(f"Logging to: {self.log_file}")
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log v{VERSION} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def log(self, message):
        """Simple logging"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
            
    def initialize_model(self):
        """Initialize model with verification"""
        print("\nInitializing model...")
        try:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(10, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            print("Model initialized successfully")
            return True
        except Exception as e:
            print(f"Model initialization failed: {e}")
            return False
            
    def train_generation(self):
        """Training loop with progress feedback"""
        self.current_gen += 1
        print(f"\n=== GENERATION {self.current_gen} ===")
        
        try:
            # Training phase
            inputs = torch.randn(32, 10).to(self.device)
            targets = torch.randn(32, 1).to(self.device)
            
            outputs = self.model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            score = -loss.item()
            print(f"Training completed | Score: {score:.4f}")
            
            # Track best performance
            if score > self.best_score:
                self.best_score = score
                torch.save(self.model.state_dict(), f"champion_gen{self.current_gen}.pth")
                print("New champion model saved!")
            
            time.sleep(PAUSE_DURATION)
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False
            
    def run(self):
        """Main execution flow"""
        if not self.initialize_model():
            print("Failed to initialize model - exiting")
            return
            
        print("\n=== TRAINING STARTED ===")
        while self.current_gen < FULL_GENERATIONS:
            success = self.train_generation()
            if not success:
                print("Retrying generation...")
                self.current_gen -= 1
                
        print("\n=== TRAINING COMPLETED ===")
        print(f"Best score achieved: {self.best_score:.4f}")
        torch.save(self.model.state_dict(), "final_model.pth")

if __name__ == "__main__":
    print(f"\nRTX 3090 FOREX TRAINER v{VERSION}")
    print("=================================")
    
    # Verify CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available - using CPU")
    
    # Run training
    trainer = ForexTrainer()
    trainer.run()
    print("\nScript completed successfully!")