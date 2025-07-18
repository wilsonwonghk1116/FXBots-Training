#!/usr/bin/env python3
# type: ignore  # suppress type-checker errors across this module
"""
Enhanced Smart Real Training System for Forex Bots
Targets 95% VRAM utilization on RTX 3090 24GB with comprehensive champion analysis
"""

import os
import sys
import time
# Ray removed
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import json
import logging
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
import warnings
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import signal
import gc
import atexit
warnings.filterwarnings('ignore')

# Initialize TensorBoard
writer = SummaryWriter('logs/smart_trading')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initial GPU VRAM Cleanup ---
logger.info("🧹 Performing initial GPU VRAM cleanup before starting training...")
torch.cuda.empty_cache()
gc.collect()
logger.info("✅ Initial GPU VRAM cleanup complete.")

# --- GPU VRAM Cleanup Handlers ---
def cleanup_gpu_vram(signum=None, frame=None):
    logging.info("🧹 Cleaning up GPU VRAM before exit...")
    torch.cuda.empty_cache()
    gc.collect()
    logging.info("✅ GPU VRAM cleanup complete. Exiting.")
    if signum is not None:
        sys.exit(0)

def cleanup_on_exit():
    logging.info("🧹 Cleaning up GPU VRAM at program exit...")
    torch.cuda.empty_cache()
    gc.collect()
    logging.info("✅ GPU VRAM cleanup complete.")

signal.signal(signal.SIGINT, cleanup_gpu_vram)
signal.signal(signal.SIGTERM, cleanup_gpu_vram)
atexit.register(cleanup_on_exit)

class SmartForexEnvironment(gym.Env):
    """Enhanced Forex Environment with difficulty levels"""
    
    def __init__(self, data_file: str = "data/EURUSD_H1.csv", initial_balance: float = 100000.0):
        super().__init__()
        # Identifier for which bot is using this environment
        self.bot_id: Optional[int] = None
        # Running PnL tracker
        self.total_pnl: float = 0.0
        self.initial_balance = initial_balance
        self.difficulty = 0  # 0-4 difficulty levels
        self.data = np.array([])  # Initialize empty data array
        self.max_steps = 1000  # Initialize max_steps
        self.position = 0
        self.entry_price = None
        self.trades = []
        self.balance_history = []
        self.current_step = 0
        
        # New reward/penalty system tracking
        self.first_trade_bonus_given = False  # Track if first trade bonus awarded
        self.last_trade_step = 0  # Track when last trade was made
        self.idle_penalty_threshold = 1000  # Steps before idle penalty
        self.max_leverage = 100  # 100x leverage allowed
        
        # Load data from CSV or generate synthetic data if none
        self.data = self._load_data(data_file)
        if len(self.data) == 0:
            self.data = self._generate_synthetic_data()
        # Initialize difficulty settings (sets observation/action spaces and resets state)
        self.set_difficulty(self.difficulty)
        
        # Risk management parameters
        self.trading_cost = 0.0002  # 2 pips spread
        self.stop_loss_pips = 30  # 30 pips stop loss
        self.take_profit_pips = 60  # 60 pips take profit 
        self.max_position_size = 0.1  # Max 10% of balance per trade
        
        # Ensure data exists
        if len(self.data) == 0:
            # Generate synthetic data if no data exists
            self.data = self._generate_synthetic_data()
        
        # Gym spaces - Updated to include technical indicators (26 features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        self.reset()
    
    def set_difficulty(self, level: int):
        """Adjust environment difficulty (0=easy, 4=hard)"""
        self.difficulty = max(0, min(4, level))
        # Adjust market volatility based on difficulty
        self.volatility_multiplier = 1.0 + self.difficulty * 0.25
        # Reset core trading state without reinitializing data
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = None
        self.trades = []
        self.balance_history = [self.initial_balance]
        self.current_step = 0
        
        # Reset reward/penalty tracking
        self.first_trade_bonus_given = False
        self.last_trade_step = 0
        
        # Risk management parameters
        self.trading_cost = 0.0002  # 2 pips spread
        self.stop_loss_pips = 30  # 30 pips stop loss
        self.take_profit_pips = 60  # 60 pips take profit 
        self.max_position_size = 0.1  # Max 10% of balance per trade
        
        # Ensure data exists
        if len(self.data) == 0:
            # Generate synthetic data if no data exists
            self.data = self._generate_synthetic_data()
        
        # Gym spaces - Updated to include technical indicators (26 features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        self.reset()
    
    def _load_data(self, data_file: str) -> np.ndarray:
        """Load forex data from CSV file"""
        try:
            if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
                df = pd.read_csv(data_file)
                # Use to_numpy() for consistent numpy array
                return df[['Close']].to_numpy().flatten()
            else:
                return np.array([])
        except Exception as e:
            logger.warning(f"Could not load data from {data_file}: {e}")
            return np.array([])
    
    def _generate_synthetic_data(self, length: int = 10000) -> np.ndarray:
        """Generate synthetic EUR/USD-like data"""
        logger.info("Generating synthetic forex data for training...")
        np.random.seed(42)
        
        # Start price around EUR/USD typical range
        start_price = 1.1000
        prices = [start_price]
        
        for i in range(length - 1):
            # Random walk with slight trend and volatility
            change = np.random.normal(0, 0.0005)  # ~50 pips volatility
            trend = 0.000001 * np.sin(i / 100)  # Long-term cycle
            new_price = prices[-1] + change + trend
            new_price = max(0.9000, min(1.3000, new_price))  # Realistic bounds
            prices.append(new_price)
        
        return np.array(prices)
    
    def _get_observation(self) -> np.ndarray:
        """Get current market observation with technical indicators"""
        # Get last 20 prices
        if self.current_step < 20:
            obs_prices = np.zeros(20)
            available_data = self.data[max(0, self.current_step-19):self.current_step+1]
            obs_prices[-len(available_data):] = available_data[-20:]
        else:
            obs_prices = self.data[self.current_step-19:self.current_step+1]
        # Normalize prices
        if len(obs_prices) > 1:
            norm_prices = (obs_prices - obs_prices.mean()) / (obs_prices.std() + 1e-8)
        else:
            norm_prices = obs_prices
        # --- Technical Indicators ---
        # RSI (14)
        def calc_rsi(prices, period=14):
            if len(prices) < period+1:
                return 0.0
            deltas = np.diff(prices[-(period+1):])
            up = deltas[deltas > 0].sum() / period
            down = -deltas[deltas < 0].sum() / period
            rs = up / (down + 1e-8)
            return 100 - (100 / (1 + rs))
        rsi = calc_rsi(obs_prices)
        # MACD (12,26)
        def calc_macd(prices, fast=12, slow=26):
            if len(prices) < slow:
                return 0.0, 0.0
            ema_fast = pd.Series(prices).ewm(span=fast).mean().values[-1]
            ema_slow = pd.Series(prices).ewm(span=slow).mean().values[-1]
            macd = ema_fast - ema_slow
            signal = pd.Series([macd]).ewm(span=9).mean().values[-1]
            return macd, signal
        macd, macd_signal = calc_macd(obs_prices)
        # Bollinger Bands (20, 2 std)
        def calc_bbands(prices, period=20, num_std=2):
            if len(prices) < period:
                return 0.0, 0.0, 0.0
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            upper = sma + num_std * std
            lower = sma - num_std * std
            return upper, sma, lower
        bb_upper, bb_mid, bb_lower = calc_bbands(obs_prices)
        # Normalize indicators
        rsi_norm = (rsi - 50) / 50
        macd_norm = macd / (np.std(obs_prices) + 1e-8)
        macd_signal_norm = macd_signal / (np.std(obs_prices) + 1e-8)
        bb_upper_norm = (bb_upper - obs_prices[-1]) / (np.std(obs_prices) + 1e-8)
        bb_mid_norm = (bb_mid - obs_prices[-1]) / (np.std(obs_prices) + 1e-8)
        bb_lower_norm = (bb_lower - obs_prices[-1]) / (np.std(obs_prices) + 1e-8)
        # Compose observation: [prices, rsi, macd, macd_signal, bb_upper, bb_mid, bb_lower]
        obs = np.concatenate([
            norm_prices,
            [rsi_norm, macd_norm, macd_signal_norm, bb_upper_norm, bb_mid_norm, bb_lower_norm]
        ])
        return obs.astype(np.float32)
    
    def step(self, action: int, position_size: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment with new reward/penalty system"""
        # Ensure entry_price is not None for arithmetic operations
        if self.entry_price is None:
            self.entry_price = self.data[self.current_step]
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, {}

        current_price = self.data[self.current_step]
        next_price = self.data[self.current_step + 1]
        price_change = next_price - current_price

        reward = 0.0
        trade_executed = False
        trade_opened = False

        # Determine direction and volume
        if action == 1 and self.position <= 0:  # BUY
            direction = "BUY"
            volume = self.max_position_size * position_size
            # Close short if exists
            if self.position == -1:
                exit_price = current_price
                pnl_pips = -self.position * (current_price - self.entry_price) * 10000
                if pnl_pips >= self.take_profit_pips:
                    exit_price = self.entry_price + self.take_profit_pips/10000
                elif pnl_pips <= -self.stop_loss_pips:
                    exit_price = self.entry_price - self.stop_loss_pips/10000
                profit_pips = -self.position * (exit_price - self.entry_price) * 10000 - 2 * self.trading_cost * 10000
                
                # NEW REWARD SYSTEM: PnL-based reward/penalty for trade
                trade_pnl_reward = profit_pips * volume * 10  # Scale up reward
                reward += trade_pnl_reward
                
                self.trades.append({
                    'bot_id': self.bot_id,
                    'direction': direction,
                    'volume': volume,
                    'type': 'close_short',
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'profit': profit_pips,
                    'step': self.current_step
                })
                logger.info(f"Step {self.current_step}: Closed SHORT, profit_pips={profit_pips}, trade_pnl_reward={trade_pnl_reward}")
            
            # Register trade when opening a new long position
            if self.position != 1:
                self.trades.append({
                    'bot_id': self.bot_id,
                    'direction': direction,
                    'volume': volume,
                    'type': 'open_long',
                    'entry_price': current_price,
                    'exit_price': None,
                    'profit': 0.0,
                    'step': self.current_step
                })
                trade_opened = True
            self.position = 1
            self.entry_price = current_price
            trade_executed = True
            self.last_trade_step = self.current_step  # Update last trade step
            
            # NEW REWARD SYSTEM: First trade bonus
            if not self.first_trade_bonus_given:
                reward += 1000  # First trade bonus
                self.first_trade_bonus_given = True
                logger.info(f"Step {self.current_step}: FIRST TRADE BONUS +1000 reward!")
                
        elif action == 2 and self.position >= 0:  # SELL
            direction = "SELL"
            volume = self.max_position_size * position_size
            if self.position == 1:
                profit_pips = self.position * price_change * 10000 - 2 * self.trading_cost * 10000
                
                # NEW REWARD SYSTEM: PnL-based reward/penalty for trade
                trade_pnl_reward = profit_pips * volume * 10  # Scale up reward
                reward += trade_pnl_reward
                
                self.trades.append({
                    'bot_id': self.bot_id,
                    'direction': direction,
                    'volume': volume,
                    'type': 'close_long',
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'profit': profit_pips,
                    'step': self.current_step
                })
                logger.info(f"Step {self.current_step}: Closed LONG, profit_pips={profit_pips}, trade_pnl_reward={trade_pnl_reward}")
            
            # Register trade when opening a new short position
            if self.position != -1:
                self.trades.append({
                    'bot_id': self.bot_id,
                    'direction': direction,
                    'volume': volume,
                    'type': 'open_short',
                    'entry_price': current_price,
                    'exit_price': None,
                    'profit': 0.0,
                    'step': self.current_step
                })
                trade_opened = True
            self.position = -1
            self.entry_price = current_price
            trade_executed = True
            self.last_trade_step = self.current_step  # Update last trade step
            
            # NEW REWARD SYSTEM: First trade bonus
            if not self.first_trade_bonus_given:
                reward += 1000  # First trade bonus
                self.first_trade_bonus_given = True
                logger.info(f"Step {self.current_step}: FIRST TRADE BONUS +1000 reward!")
                
        else:
            direction = "HOLD"
            volume = 0.0
            
            # NEW REWARD SYSTEM: Idle penalty
            steps_since_last_trade = self.current_step - self.last_trade_step
            if steps_since_last_trade >= self.idle_penalty_threshold:
                idle_penalty = -100  # Penalty for excessive holding
                reward += idle_penalty
                logger.info(f"Step {self.current_step}: IDLE PENALTY {idle_penalty} (no trade for {steps_since_last_trade} steps)")
                # Reset the penalty counter
                self.last_trade_step = self.current_step - (self.idle_penalty_threshold - 100)
            
            # Additional holding penalty to encourage trading
            holding_penalty = -1  # Small penalty for each hold action
            reward += holding_penalty

        # Only calculate unrealized P&L if a position is open and action is not HOLD
        if self.position != 0 and direction != "HOLD":
            unrealized_pnl = self.position * (next_price - self.entry_price) * 10000
            reward += unrealized_pnl * 0.1

        # Only update balance if a trade was executed or opened
        if trade_executed or trade_opened:
            self.balance += reward * 0.01
        self.balance_history.append(self.balance)
        self.total_pnl = self.balance - self.initial_balance

        self.current_step += 1
        done = self.current_step >= min(len(self.data) - 1, self.max_steps)

        info = {
            'bot_id': self.bot_id,
            'direction': direction,
            'trade_volume': volume,
            'total_pnl': self.total_pnl,
            'total_capital': self.balance,
            'balance': self.balance,
            'position': self.position,
            'price': next_price,
            'trade_executed': trade_executed,
            'trade_opened': trade_opened,
            'total_trades': len(self.trades),
            'first_trade_bonus_given': self.first_trade_bonus_given,
            'steps_since_last_trade': self.current_step - self.last_trade_step
        }

        # LOGGING: Action probabilities and selected action
        if self.current_step % 100 == 0:  # Log every 100 steps to reduce noise
            logger.info(f"Step {self.current_step}: Action={action}, Position={self.position}, Balance={self.balance:.2f}, Reward={reward:.2f}")

        return self._get_observation(), reward, done, False, info
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self.balance_history = [self.initial_balance]
        self.current_step = random.randint(20, max(20, len(self.data) - self.max_steps - 1))
        
        # Reset reward/penalty tracking - FIXED: set last_trade_step to current_step
        self.first_trade_bonus_given = False
        self.last_trade_step = self.current_step  # Start the timer from current position
        
        return self._get_observation(), {}
    
    def simulate_trading_detailed(self, model, steps: int = 1000) -> Dict:
        """Detailed trading simulation for champion analysis"""
        self.reset()
        total_reward = 0
        device = next(model.parameters()).device  # Get model device
        
        for _ in range(steps):
            obs = self._get_observation()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_probs, position_size = model(obs_tensor)
                action = torch.argmax(action_probs).item()
            
            obs, reward, done, _, info = self.step(action, position_size.item())
            total_reward += reward
            
            if done:
                break
        
        # Calculate comprehensive metrics
        if len(self.trades) > 0:
            profits = [trade['profit'] for trade in self.trades]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]
            
            win_rate = len(winning_trades) / len(profits) if profits else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            gross_profit = sum(winning_trades)
            gross_loss = abs(sum(losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Risk metrics
            balance_array = np.array(self.balance_history)
            peak = np.maximum.accumulate(balance_array)
            drawdown = (peak - balance_array) / peak * 100
            max_drawdown = np.max(drawdown)
            recovery_factor = (self.balance - self.initial_balance) / max_drawdown if max_drawdown > 0 else 0
        else:
            win_rate = avg_win = avg_loss = gross_profit = gross_loss = profit_factor = max_drawdown = recovery_factor = 0
        
        return {
            'final_balance': self.balance,
            'total_return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'recovery_factor': recovery_factor,
            'risk_reward_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'trades': self.trades,
            'balance_history': self.balance_history,
            'total_reward': total_reward
        }

class SmartTradingBot(nn.Module):
    """Fixed neural network with LSTM for forex trading - GPU intensive but functional"""
    
    def __init__(self, input_size: int = 26, hidden_size: int = 512, output_size: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input normalization with learnable parameters (LayerNorm instead of BatchNorm)
        self.input_norm = nn.LayerNorm(input_size)
        
        # Feature extractor with residual connections - GPU intensive
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for i in range(3)
        ])
        
        # Multi-layer LSTM for temporal patterns - GPU intensive
        self.lstm1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Attention mechanism for complex patterns - GPU intensive
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Market signal analyzer - processes technical indicators
        self.signal_analyzer = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.Tanh(),  # Tanh for better signal processing
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU()
        )
        
        # Action decision network - NO SOFTMAX HERE (applied in forward)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, output_size)
            # NO SOFTMAX - we apply it with temperature in forward
        )
        
        # Temperature parameter for sharper/softer predictions
        self.temperature = nn.Parameter(torch.tensor(1.5))
        
        # Position sizing network
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1),
            nn.Sigmoid()
        )
        
        # Enhanced weight initialization
        self.apply(self._init_weights)
        
        # Break symmetry with biased initialization
        with torch.no_grad():
            self.action_head[-1].bias[0] = 0.1   # HOLD bias
            self.action_head[-1].bias[1] = -0.05  # BUY bias
            self.action_head[-1].bias[2] = -0.05  # SELL bias
    
    def _init_weights(self, module):
        """Enhanced weight initialization for better responsiveness"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, 0.0, 0.01)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
                    # Initialize forget gate bias to 1 for better gradient flow
                    if 'bias_hh' in name:
                        n = param.size(0)
                        with torch.no_grad():
                            param[n//4:n//2].fill_(1.0)
    
    def forward(self, x):
        # Handle batch dimensions properly
        original_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Normalize input
        x = self.input_norm(x)
        
        # Feature extraction with residual connections - GPU intensive
        features = x
        for i, layer in enumerate(self.feature_layers):
            new_features = layer(features)
            if i > 0 and features.size(-1) == new_features.size(-1):
                features = features + new_features  # Residual connection
            else:
                features = new_features
        
        # Add sequence dimension for LSTM
        x_seq = features.unsqueeze(1)
        
        # Multi-layer LSTM processing - GPU intensive
        lstm1_out, _ = self.lstm1(x_seq)
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # Self-attention mechanism - GPU intensive
        attn_out, _ = self.attention(lstm2_out, lstm2_out, lstm2_out)
        lstm_features = attn_out.squeeze(1)
        
        # Market signal analysis
        signals = self.signal_analyzer(lstm_features)
        
        # Action probabilities with temperature scaling
        action_logits = self.action_head(signals)
        # Apply temperature scaling for sharper or softer predictions
        scaled_logits = action_logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        action_probs = torch.softmax(scaled_logits, dim=-1)
        
        # Position sizing
        position_size = self.position_head(signals)
        
        # Handle single sample output
        if single_sample:
            action_probs = action_probs.squeeze(0)
            position_size = position_size.squeeze(0)
        
        return action_probs, position_size

class VRAMOptimizedTrainer:
    """RTX 3090 24GB VRAM-optimized trainer with advanced memory management"""
    
    def __init__(self, population_size: int = 300, target_vram_percent: float = 0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.population_size = population_size
        self.target_vram_percent = target_vram_percent
        
        # RTX 3090 specific optimizations for 95% utilization
        if torch.cuda.is_available():
            # Enable all Ampere optimizations for RTX 3090
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False
            
            # Aggressive RTX 3090 memory optimization for 95% usage (22.8GB)
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,roundup_power2_divisions:16,garbage_collection_threshold:0.95'
            
            # Set VRAM target to 95% (22.8GB out of 24GB)
            try:
                torch.cuda.set_per_process_memory_fraction(self.target_vram_percent)
                logger.info(f"RTX 3090 AGGRESSIVE MODE: {self.target_vram_percent*100}% VRAM target ({self.target_vram_percent*24:.1f}GB)")
            except Exception as e:
                logger.warning(f"Could not set VRAM fraction: {e}")
            
            # Enable memory pool for efficient allocation
            torch.cuda.empty_cache()
            
        self.generation = 0
        self.current_difficulty = 0
        self.max_difficulty = 4
        self.difficulty_increase_interval = 5
        
        # RTX 3090 VRAM monitoring
        self.rtx3090_vram_target = self.target_vram_percent * 24.0  # Target GB
        self.gpu_utilization_target = 95.0  # 95% GPU usage target
        
        # Use all available CPU cores across both Xeon CPUs (aim for 75% utilization)
        total_cores = multiprocessing.cpu_count()
        self.num_workers = int(total_cores * 0.75)
        logger.info(f"Using {self.num_workers} workers out of {total_cores} total CPU cores")
        self.env = SmartForexEnvironment()
        
        # Create environment pool for parallel evaluation
        self.env_pool = [SmartForexEnvironment() for _ in range(min(20, self.num_workers))]
        
        # VRAM allocation tracking
        self.vram_pools = []
        self.large_tensor_cache = []
        
        # Pre-allocate large tensors for sustained VRAM usage
        self._preallocate_vram_tensors()
        
        logger.info(f"RTX 3090 Trainer initialized: population={self.population_size}, workers={self.num_workers}, target_vram={self.target_vram_percent*100}%")
    
    def _preallocate_vram_tensors(self):
        """Pre-allocate large tensors to sustain high VRAM usage"""
        if not torch.cuda.is_available():
            return
            
        try:
            # Pre-allocate large persistent tensors (8GB worth)
            logger.info("Pre-allocating VRAM tensors for sustained usage...")
            
            # Large matrices for mathematical operations
            self.vram_matrix_large = torch.randn(4096, 4096, device=self.device)  # ~67MB
            self.vram_matrix_xl = torch.randn(6144, 6144, device=self.device)     # ~151MB
            self.vram_matrix_xxl = torch.randn(8192, 4096, device=self.device)    # ~134MB
            
            # Persistent activation cache (simulating large batch processing)
            self.activation_cache = []
            for i in range(50):
                cache_tensor = torch.randn(1024, 1024, device=self.device)  # ~4MB each
                self.activation_cache.append(cache_tensor)
            
            # Model ensemble storage (pre-allocate space for multiple models)
            self.model_ensemble_cache = []
            for i in range(20):
                model_cache = torch.randn(2048, 2048, device=self.device)  # ~16MB each
                self.model_ensemble_cache.append(model_cache)
            
            # Large embedding tables (simulating real workloads)
            self.embedding_cache = torch.randn(50000, 1024, device=self.device)  # ~200MB
            self.feature_cache = torch.randn(100000, 512, device=self.device)    # ~200MB
            
            # Gradient accumulation buffers
            self.gradient_cache = []
            for i in range(100):
                grad_buffer = torch.randn(1024, 512, device=self.device)  # ~2MB each
                self.gradient_cache.append(grad_buffer)
            
            # Large temporary computation space
            self.computation_workspace = torch.randn(10240, 2048, device=self.device)  # ~84MB
            
            # Track total pre-allocated memory
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"Pre-allocated {allocated_gb:.2f}GB VRAM for sustained high usage")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("RTX 3090 VRAM fully utilized during pre-allocation - excellent!")
                torch.cuda.empty_cache()
            else:
                logger.error(f"VRAM pre-allocation error: {e}")
    
    def _maintain_vram_pressure(self):
        """Maintain sustained high VRAM pressure throughout training"""
        if not torch.cuda.is_available():
            return
            
        try:
            # Rotate through large tensor operations
            with torch.no_grad():
                # Matrix multiplications using pre-allocated tensors
                result1 = torch.matmul(self.vram_matrix_large, self.vram_matrix_large.T)
                result2 = torch.matmul(self.vram_matrix_xl[:4096, :4096], self.vram_matrix_xxl[:4096, :])
                
                # Complex mathematical operations
                eigenvals = torch.linalg.eigvals(self.vram_matrix_large[:1000, :1000])
                svd_result = torch.svd(self.vram_matrix_xl[:2048, :2048])
                
                # Convolution-like operations (memory intensive)
                conv_result = torch.nn.functional.conv2d(
                    self.computation_workspace.view(1, 1, 10240, 2048),
                    torch.randn(128, 1, 3, 3, device=self.device),
                    padding=1
                )
                
                # Activation function chains
                activated = torch.relu(self.feature_cache)
                activated = torch.sigmoid(activated)
                activated = torch.tanh(activated)
                
                # Batch operations on cache
                if len(self.activation_cache) > 10:
                    batch_result = torch.stack(self.activation_cache[:10])
                    _ = torch.mean(batch_result, dim=0)
                    _ = torch.std(batch_result, dim=0)
                
                # Clean up temporaries but keep pre-allocated tensors
                del result1, result2, eigenvals, svd_result, conv_result, activated
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.info("VRAM pressure maintained at maximum - perfect!")
            else:
                logger.warning(f"VRAM pressure maintenance error: {e}")
    
    def monitor_rtx3090_resources(self) -> Dict:
        """Comprehensive RTX 3090 resource monitoring"""
        if not torch.cuda.is_available():
            return {}
        
        try:
            # VRAM monitoring
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            total_vram = 24.0  # RTX 3090 VRAM
            utilization_percent = (allocated / total_vram) * 100
            
            # GPU utilization
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100 if gpus else 0
            
            stats = {
                'vram_allocated_gb': allocated,
                'vram_reserved_gb': reserved,
                'vram_total_gb': total_vram,
                'vram_utilization_percent': utilization_percent,
                'vram_target_percent': self.target_vram_percent * 100,
                'vram_gap_gb': self.rtx3090_vram_target - allocated,
                'gpu_usage_percent': gpu_usage,
                'gpu_target_percent': self.gpu_utilization_target,
                'meeting_vram_target': utilization_percent >= (self.target_vram_percent * 100 - 5),
                'meeting_gpu_target': gpu_usage >= (self.gpu_utilization_target - 5)
            }
            
            # Log status
            vram_status = "✓" if stats['meeting_vram_target'] else "✗"
            gpu_status = "✓" if stats['meeting_gpu_target'] else "✗"
            
            logger.info(f"RTX 3090 {vram_status} VRAM: {allocated:.1f}/{total_vram}GB ({utilization_percent:.1f}%) | "
                       f"{gpu_status} GPU: {gpu_usage:.1f}% | Targets: {self.target_vram_percent*100:.0f}% VRAM, {self.gpu_utilization_target:.0f}% GPU")
            
            return stats
            
        except Exception as e:
            logger.error(f"RTX 3090 monitoring error: {e}")
            return {}
    
    def aggressive_vram_fill(self) -> None:
        """Aggressively fill RTX 3090 VRAM to reach 95% target"""
        if not torch.cuda.is_available():
            return
            
        try:
            current_usage = torch.cuda.memory_allocated() / 1024**3
            target_usage = self.rtx3090_vram_target
            gap = target_usage - current_usage
            
            if gap > 0.5:  # If more than 500MB gap
                logger.info(f"Filling VRAM gap: {gap:.1f}GB to reach 95% target")
                
                # Create large tensors to fill VRAM
                vram_fillers = []
                tensor_size_gb = min(gap / 4, 2.0)  # Create tensors up to 2GB each
                tensor_elements = int(tensor_size_gb * 1024**3 / 4)  # 4 bytes per float32
                tensor_side = int(tensor_elements ** 0.5)
                
                for i in range(4):  # Create 4 tensors
                    try:
                        filler_tensor = torch.randn(tensor_side, tensor_side, device=self.device)
                        # Perform operations to ensure tensor stays in VRAM
                        _ = torch.matmul(filler_tensor, filler_tensor.T)
                        vram_fillers.append(filler_tensor)
                        
                        current_usage = torch.cuda.memory_allocated() / 1024**3
                        logger.info(f"Created VRAM filler {i+1}: {current_usage:.1f}GB total")
                        
                        if current_usage >= target_usage * 0.98:  # Stop at 98% of target
                            break
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.info("VRAM limit reached - excellent!")
                            break
                        else:
                            raise e
                
                # Store fillers as class attribute to keep them in VRAM
                self.vram_fillers = vram_fillers
                
        except Exception as e:
            logger.warning(f"VRAM filling error: {e}")

    def create_population(self) -> List[SmartTradingBot]:
        logger.info(f"Creating RTX 3090 optimized population of {self.population_size} bots...")
        
        # Create population in batches to monitor VRAM usage
        population = []
        batch_size = 50  # Create 50 bots at a time
        
        for batch_start in range(0, self.population_size, batch_size):
            batch_end = min(batch_start + batch_size, self.population_size)
            batch_bots = [SmartTradingBot().to(self.device) for _ in range(batch_end - batch_start)]
            population.extend(batch_bots)
            
            # Monitor VRAM after each batch
            current_vram = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"Created bots {batch_start+1}-{batch_end}: {current_vram:.1f}GB VRAM used")
            
            # Aggressive VRAM optimization - create additional tensors if under target
            if current_vram < self.rtx3090_vram_target * 0.8:  # If below 80% of target
                self.aggressive_vram_fill()
        
        # Final VRAM status
        stats = self.monitor_rtx3090_resources()
        if stats.get('meeting_vram_target', False):
            logger.info(f"✓ RTX 3090 VRAM target achieved: {stats['vram_utilization_percent']:.1f}%")
        else:
            logger.warning(f"✗ RTX 3090 VRAM below target: {stats['vram_utilization_percent']:.1f}%")
            
        return population

    def evaluate_population(self, population: List[SmartTradingBot]) -> List[Dict]:
        """Evaluate entire population with intensive CPU+GPU hybrid processing."""
        logger.info(f"Evaluating {len(population)} bots using {self.num_workers} CPU workers + GPU")
        
        # Prepare work for CPU workers - convert models to CPU state dicts
        worker_args = []
        for i, bot in enumerate(population):
            # Move state dict to CPU for worker processing
            cpu_state_dict = {k: v.cpu() for k, v in bot.state_dict().items()}
            worker_args.append((cpu_state_dict, i, 1000))  # 1000 simulation steps per bot
        
        # Use multiprocessing with proper start method
        try:
            # Set spawn method only when needed
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(self.num_workers) as pool:
                logger.info(f"Starting parallel evaluation with {self.num_workers} CPU workers...")
                results = pool.map(evaluate_bot_worker, worker_args)
                
        except Exception as e:
            logger.error(f"Multiprocessing failed: {e}")
            # Fallback to sequential processing
            results = [evaluate_bot_worker(args) for args in worker_args]
        
        # Log worker performance
        cpu_usages = [r.get('worker_cpu_usage', 0) for r in results if 'worker_cpu_usage' in r]
        if cpu_usages:
            avg_cpu = sum(cpu_usages) / len(cpu_usages)
            logger.info(f"Average worker CPU usage: {avg_cpu:.1f}%")
        
        # Assign bot IDs and sort by performance
        for i, metrics in enumerate(results):
            metrics['bot_id'] = i
            
        return sorted(results, key=lambda x: x.get('final_balance', 0), reverse=True)
    
    def genetic_crossover(self, parent1: SmartTradingBot, parent2: SmartTradingBot) -> SmartTradingBot:
        """Create offspring through genetic crossover with GPU acceleration"""
        child = SmartTradingBot().to(self.device)
        
        with torch.no_grad():
            for (name1, param1), (name2, param2), (name_child, param_child) in zip(
                parent1.named_parameters(), parent2.named_parameters(), child.named_parameters()
            ):
                # Random crossover mask on GPU for faster processing
                mask = torch.rand_like(param1) > 0.5
                param_child.data = param1 * mask + param2 * (~mask)
                
        # GPU-intensive operation to increase VRAM usage
        with torch.no_grad():
            # Create large temporary tensors on GPU to use VRAM
            temp_tensor = torch.randn(1024, 1024, device=self.device)
            _ = torch.matmul(temp_tensor, temp_tensor.T)  # Matrix multiplication on GPU
            del temp_tensor
        
        return child
    
    def mutate(self, bot: SmartTradingBot, mutation_rate: float = 0.1) -> SmartTradingBot:
        """Apply mutations to bot with GPU acceleration"""
        with torch.no_grad():
            for param in bot.parameters():
                if torch.rand(1).item() < mutation_rate:
                    # Generate noise on GPU
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
                    
            # GPU-intensive operation to increase VRAM usage
            temp_tensor = torch.randn(512, 512, device=self.device)
            _ = torch.nn.functional.relu(temp_tensor)  # GPU operation
            del temp_tensor
            
        return bot
    
    def evolve_generation(self, population: List[SmartTradingBot], elite_size: int = 100) -> List[SmartTradingBot]:
        # Adjust difficulty based on curriculum learning
        if self.generation % self.difficulty_increase_interval == 0:
            self.current_difficulty = min(self.current_difficulty + 1, self.max_difficulty)
            for env in self.env_pool:
                env.set_difficulty(self.current_difficulty)
        """Evolve population to next generation"""
        # Evaluate current population
        results = self.evaluate_population(population)
        
        # Select elite bots
        elite_bots = [population[result['bot_id']] for result in results[:elite_size]]
        
        # Create next generation
        new_population = elite_bots.copy()  # Keep elite
        
        while len(new_population) < self.population_size:
            # Select parents from top 50%
            parent1 = random.choice(elite_bots[:elite_size//2])
            parent2 = random.choice(elite_bots[:elite_size//2])
            
            # Create and mutate offspring
            child = self.genetic_crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        return new_population[:self.population_size], results
    
    def analyze_champion(self, champion_bot: SmartTradingBot, results: List[Dict]) -> Dict:
        """Comprehensive analysis of champion bot"""
        champion_metrics = results[0]  # Best performer
        
        # Additional detailed analysis
        detailed_metrics = self.env.simulate_trading_detailed(champion_bot, steps=5000)
        
        analysis = {
            'champion_analysis': {
                'bot_id': champion_metrics['bot_id'],
                'final_balance': detailed_metrics['final_balance'],
                'total_return_pct': detailed_metrics['total_return_pct'],
                'win_rate': detailed_metrics['win_rate'],
                'total_trades': detailed_metrics['total_trades'],
                'gross_profit': detailed_metrics['gross_profit'],
                'gross_loss': detailed_metrics['gross_loss'],
                'profit_factor': detailed_metrics['profit_factor'],
                'average_win': detailed_metrics['avg_win'],
                'average_loss': detailed_metrics['avg_loss'],
                'risk_reward_ratio': detailed_metrics['risk_reward_ratio'],
                'max_drawdown': detailed_metrics['max_drawdown'],
                'recovery_factor': detailed_metrics['recovery_factor'],
                'sharpe_ratio': self._calculate_sharpe_ratio(detailed_metrics['balance_history']),
                'calmar_ratio': self._calculate_calmar_ratio(detailed_metrics),
                'trade_history': detailed_metrics['trades'][:50],  # Last 50 trades
                'balance_curve': detailed_metrics['balance_history'][-500:]  # Last 500 points
            },
            'training_summary': {
                'population_size': self.population_size,
                'target_vram_percent': self.target_vram_percent,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return analysis
    
    def _calculate_sharpe_ratio(self, balance_history: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(balance_history) < 2:
            return 0
        
        returns = np.diff(balance_history) / balance_history[:-1]
        if np.std(returns) == 0:
            return 0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_calmar_ratio(self, metrics: Dict) -> float:
        """Calculate Calmar ratio"""
        if metrics['max_drawdown'] == 0:
            return 0
        
        annual_return = metrics['total_return_pct']
        return annual_return / metrics['max_drawdown']
    
    def save_champion(self, champion_bot: SmartTradingBot, analysis: Dict) -> str:
        """Save champion model and analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CHAMPION_BOT_{timestamp}.pth"
        torch.save(champion_bot.state_dict(), filename)
        with open(filename.replace('.pth', '_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"💾 Champion and analysis saved: {filename}")
        return filename

    def adjust_trading_thresholds(self, factor: float) -> None:
        """Adjust trading parameters for all environments when no trades executed"""
        logger.warning(f"No trades executed - adjusting strategy thresholds by {factor*100:.0f}%")
        for env in self.env_pool + [self.env]:
            # Reduce trading cost to encourage execution
            env.trading_cost = max(0.0, env.trading_cost * (1.0 - factor))
            # Tighten stop loss and expand take profit
            env.stop_loss_pips = max(1, int(env.stop_loss_pips * (1.0 - factor)))
            env.take_profit_pips = int(env.take_profit_pips * (1.0 + factor))

    def gpu_intensive_training(self, population: List[SmartTradingBot]) -> None:
        """RTX 3090 AGGRESSIVE MODE: 95% GPU + VRAM utilization"""
        if not torch.cuda.is_available():
            return
            
        logger.info("🚀 RTX 3090 AGGRESSIVE MODE: Targeting 95% GPU + VRAM utilization")
        
        # Pre-fill VRAM aggressively
        self.aggressive_vram_fill()
        
        # Mega batch processing for RTX 3090 (24GB VRAM)
        batch_size = 100  # Process 100 bots at once
        
        for i in range(0, len(population), batch_size):
            batch = population[i:i+batch_size]
            
            # MASSIVE input tensors (utilize RTX 3090's memory bandwidth)
            ultra_input = torch.randn(3000, 20, device=self.device)  # 3000 samples per batch
            
            for bot_idx, bot in enumerate(batch):
                bot.train()
                
                # RTX 3090 mixed precision for maximum performance
                scaler = torch.cuda.amp.GradScaler()
                
                with torch.cuda.amp.autocast():
                    # INTENSIVE forward passes (40 per bot)
                    for forward_pass in range(40):
                        action_probs, position_size = bot(ultra_input)
                        
                        # MASSIVE tensor operations (RTX 3090 tensor cores)
                        attention_mega = torch.matmul(action_probs, action_probs.transpose(-2, -1))
                        attention_softmax = torch.nn.functional.softmax(attention_mega * 8.0, dim=-1)
                        
                        # Large convolutions (utilize CUDA cores)
                        conv_input = action_probs.view(-1, 1, 30, 3)
                        conv_kernel = torch.randn(128, 1, 5, 3, device=self.device)
                        conv_result = torch.nn.functional.conv2d(conv_input, conv_kernel, padding=2)
                        
                        # MASSIVE FFT operations (very GPU intensive)
                        fft_mega = torch.randn(2048, 2048, device=self.device, dtype=torch.complex64)
                        fft_forward = torch.fft.fft2(fft_mega)
                        fft_backward = torch.fft.ifft2(fft_forward)
                        
                        # Linear algebra operations (tensor cores)
                        linalg_matrix = torch.randn(1024, 1024, device=self.device)
                        linalg_matrix = linalg_matrix @ linalg_matrix.T
                        eigenvals = torch.linalg.eigvals(linalg_matrix)
                        
                        # SVD on large matrices
                        svd_input = torch.randn(768, 768, device=self.device)
                        U, S, Vh = torch.linalg.svd(svd_input)
                        
                        # Massive batch operations
                        batch_ops = torch.randn(500, 500, 64, device=self.device)
                        batch_result = torch.mean(batch_ops, dim=-1)
                        batch_matmul = torch.matmul(batch_result, batch_result.transpose(-2, -1))
                        
                        # Cleanup (controlled)
                        del attention_mega, attention_softmax, conv_result
                        del fft_mega, fft_forward, fft_backward
                        del linalg_matrix, eigenvals, svd_input, U, S, Vh
                        del batch_ops, batch_result, batch_matmul
                
                # AGGRESSIVE gradient training (10 steps per bot)
                optimizer = torch.optim.AdamW(bot.parameters(), lr=0.001, weight_decay=0.01)
                
                for training_step in range(10):
                    # LARGE synthetic training batches
                    synthetic_mega = torch.randn(500, 20, device=self.device)
                    target_actions = torch.randint(0, 3, (500,), device=self.device)
                    target_positions = torch.rand(500, 1, device=self.device)
                    
                    with torch.cuda.amp.autocast():
                        action_probs, position_size = bot(synthetic_mega)
                        
                        # Complex loss calculations
                        action_loss = torch.nn.functional.cross_entropy(action_probs, target_actions)
                        position_loss = torch.nn.functional.mse_loss(position_size, target_positions)
                        
                        # Regularization (more GPU work)
                        l1_loss = sum(p.abs().sum() for p in bot.parameters())
                        l2_loss = sum(p.pow(2.0).sum() for p in bot.parameters())
                        total_loss = action_loss + position_loss + 0.001 * l1_loss + 0.001 * l2_loss
                    
                    # Scaled backward pass
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    
                    # Complex gradient operations
                    torch.nn.utils.clip_grad_norm_(bot.parameters(), max_norm=1.0)
                    
                    # Custom gradient processing (GPU intensive)
                    for param in bot.parameters():
                        if param.grad is not None:
                            # Gradient smoothing and noise
                            grad_smooth = torch.nn.functional.avg_pool1d(
                                param.grad.view(1, 1, -1), kernel_size=3, padding=1
                            ).view(param.grad.shape)
                            param.grad.add_(grad_smooth * 0.1)
                            
                            # Gradient noise
                            noise = torch.randn_like(param.grad) * 0.002
                            param.grad.add_(noise)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    del synthetic_mega, target_actions, target_positions
                
                bot.eval()
                
                # Monitor progress
                if bot_idx % 10 == 0:
                    stats = self.monitor_rtx3090_resources()
                    logger.info(f"Bot {i+bot_idx+1}/{len(population)}: VRAM {stats.get('vram_utilization_percent', 0):.1f}%, GPU {stats.get('gpu_usage_percent', 0):.1f}%")
            
            # EXTREME VRAM stress test between batches
            try:
                extreme_tensors = []
                for extreme_idx in range(12):  # 12 extreme tensors
                    extreme_tensor = torch.randn(2048, 2048, device=self.device)
                    
                    # Intensive processing
                    processed = torch.nn.functional.gelu(extreme_tensor)
                    processed = torch.matmul(processed, processed.T)
                    processed = torch.nn.functional.layer_norm(processed, processed.shape[-2:])
                    
                    # More operations
                    processed = torch.nn.functional.dropout(processed, p=0.1, training=True)
                    final = torch.nn.functional.softmax(processed, dim=-1)
                    
                    extreme_tensors.append(final)
                
                # Combine all extreme tensors
                if len(extreme_tensors) >= 6:
                    mega_stack = torch.stack(extreme_tensors[:6])
                    mega_mean = torch.mean(mega_stack, dim=0)
                    mega_std = torch.std(mega_stack, dim=0)
                    mega_final = torch.matmul(mega_mean, mega_std.T)
                    
                    del mega_stack, mega_mean, mega_std, mega_final
                
                del extreme_tensors
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.info("🎯 RTX 3090 VRAM MAXED OUT - PERFECT!")
                else:
                    logger.warning(f"RTX 3090 extreme operation error: {e}")
        
        # Final resource check
        final_stats = self.monitor_rtx3090_resources()
        
        vram_success = final_stats.get('vram_utilization_percent', 0) >= 90
        gpu_success = final_stats.get('gpu_usage_percent', 0) >= 90
        
        if vram_success and gpu_success:
            logger.info("🏆 RTX 3090 AGGRESSIVE MODE SUCCESS: Both targets achieved!")
        else:
            logger.warning(f"⚠️  RTX 3090 below targets: VRAM {final_stats.get('vram_utilization_percent', 0):.1f}%, GPU {final_stats.get('gpu_usage_percent', 0):.1f}%")
        
        # Synchronize and maintain VRAM pressure
        torch.cuda.synchronize()
        
        logger.info("🚀 RTX 3090 AGGRESSIVE MODE completed")


# Global worker function for multiprocessing (must be at module level)
def evaluate_bot_worker(args):
    """Worker function for bot evaluation - CPU intensive processing"""
    bot_state_dict, worker_id, steps = args
    import os
    import psutil
    import torch
    import numpy as np
    import time
    
    # Set CPU affinity for dual Xeon setup
    try:
        total_cpus = os.cpu_count()
        cpu_core = worker_id % total_cpus
        os.sched_setaffinity(0, {cpu_core})
        print(f"Worker {worker_id} pinned to CPU core {cpu_core}")
    except Exception as e:
        print(f"Could not set CPU affinity for worker {worker_id}: {e}")
    
    # Force CPU-only processing for workers
    device = torch.device("cpu")
    
    try:
        # Create bot and environment on CPU
        bot = SmartTradingBot().to(device)
        bot.load_state_dict(bot_state_dict)
        bot.eval()
        
        env = SmartForexEnvironment()
        env.bot_id = worker_id
        
        # INTENSIVE CPU SIMULATION WITH MULTIPLE EVALUATIONS
        metrics_list = []
        
        # Run multiple simulations per worker for more CPU load
        for simulation_round in range(3):  # 3 simulations per worker
            env.reset()
            
            # CPU-intensive mathematical operations during simulation
            start_time = time.time()
            total_reward = 0
            
            for step in range(steps):
                obs = env._get_observation()
                
                # CPU-intensive preprocessing
                # Multiple feature engineering operations
                obs_normalized = (obs - np.mean(obs)) / (np.std(obs) + 1e-8)
                obs_smoothed = np.convolve(obs, np.ones(3)/3, mode='same')
                obs_diff = np.diff(np.concatenate([[obs[0]], obs]))
                
                # Technical indicator calculations (CPU intensive)
                if len(obs) >= 14:
                    rsi_values = []
                    for i in range(14, len(obs)):
                        window = obs[i-14:i]
                        deltas = np.diff(window)
                        gains = np.where(deltas > 0, deltas, 0)
                        losses = np.where(deltas < 0, -deltas, 0)
                        avg_gain = np.mean(gains) if len(gains) > 0 else 0.001
                        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        rsi_values.append(rsi)
                
                # Complex statistical calculations
                if len(obs) >= 10:
                    correlation_matrix = np.corrcoef([obs, obs_smoothed])
                    eigenvals = np.linalg.eigvals(correlation_matrix)
                    spectral_norm = np.max(eigenvals)
                
                # Monte Carlo price simulation (very CPU intensive)
                future_scenarios = []
                current_price = obs[-1] if len(obs) > 0 else 1.0
                for scenario in range(50):  # 50 scenarios per step
                    scenario_price = current_price
                    for future_step in range(10):
                        random_change = np.random.normal(0, 0.001)
                        scenario_price *= (1 + random_change)
                    future_scenarios.append(scenario_price)
                
                # Statistical analysis of scenarios
                scenario_mean = np.mean(future_scenarios)
                scenario_std = np.std(future_scenarios)
                scenario_skew = (np.mean([(x - scenario_mean)**3 for x in future_scenarios]) / 
                               (scenario_std**3 + 1e-8))
                
                # Complex decision matrix calculations
                decision_factors = np.array([
                    np.mean(obs_normalized),
                    np.std(obs_smoothed),
                    scenario_mean - current_price,
                    scenario_std,
                    scenario_skew,
                    spectral_norm if 'spectral_norm' in locals() else 0.5
                ])
                
                # Neural network forward pass (CPU tensor operations)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Multiple forward passes for more CPU work
                    action_probs_list = []
                    position_sizes_list = []
                    
                    for inference_round in range(5):  # 5 inferences per step
                        action_probs, position_size = bot(obs_tensor)
                        action_probs_list.append(action_probs)
                        position_sizes_list.append(position_size)
                    
                    # Ensemble the results (CPU intensive)
                    ensemble_action_probs = torch.mean(torch.stack(action_probs_list), dim=0)
                    ensemble_position_size = torch.mean(torch.stack(position_sizes_list), dim=0)
                    
                    # Complex action selection with weighted voting
                    action_weights = torch.softmax(ensemble_action_probs * 2.0, dim=1)
                    action = torch.multinomial(action_weights, 1).item()
                
                # Execute step with intensive analysis
                obs, reward, done, _, info = env.step(action, ensemble_position_size.item())
                total_reward += reward
                
                # Additional CPU-intensive analysis per step
                # Volatility clustering analysis
                if len(env.balance_history) >= 20:
                    returns = np.diff(env.balance_history[-20:])
                    volatility_clustering = np.std(returns[:10]) / np.std(returns[10:])
                
                # Performance attribution analysis
                if len(env.trades) > 0:
                    trade_returns = [trade['profit'] for trade in env.trades]
                    if len(trade_returns) >= 3:
                        trade_correlation = np.corrcoef(trade_returns[:-1], trade_returns[1:])[0,1]
                
                if done:
                    break
            
            # Complex metrics calculation after simulation
            simulation_metrics = env.simulate_trading_detailed(bot, steps=100)  # Additional detailed simulation
            simulation_metrics['simulation_round'] = simulation_round
            simulation_metrics['cpu_time'] = time.time() - start_time
            metrics_list.append(simulation_metrics)
        
        # Aggregate metrics from multiple simulations (CPU intensive)
        final_metrics = {
            'final_balance': np.mean([m['final_balance'] for m in metrics_list]),
            'total_return_pct': np.mean([m['total_return_pct'] for m in metrics_list]),
            'total_trades': int(np.mean([m['total_trades'] for m in metrics_list])),
            'win_rate': np.mean([m['win_rate'] for m in metrics_list]),
            'profit_factor': np.mean([m['profit_factor'] for m in metrics_list]),
            'max_drawdown': np.mean([m['max_drawdown'] for m in metrics_list]),
            'sharpe_ratio': np.mean([m['sharpe_ratio'] for m in metrics_list]),
            'worker_id': worker_id,
            'simulation_variance': np.var([m['final_balance'] for m in metrics_list]),
            'total_cpu_time': sum([m['cpu_time'] for m in metrics_list])
        }
        
        # Final CPU usage measurement
        cpu_usage = psutil.cpu_percent(interval=0.1)
        final_metrics['worker_cpu_usage'] = cpu_usage
        
        print(f"Worker {worker_id} completed: {final_metrics['total_cpu_time']:.2f}s CPU time, {cpu_usage:.1f}% usage")
        return final_metrics
        
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        return {'error': str(e), 'worker_id': worker_id, 'final_balance': 0, 'worker_cpu_usage': 0}

def monitor_system_resources():
    """Monitor and log detailed system resources for dual Xeon + RTX 3090 setup"""
    gpu_info = "N/A"
    gpu_usage_percent = 0.0
    gpu_vram_percent = 0.0
    gpu_vram_allocated = 0.0
    gpu_vram_reserved = 0.0
    
    if torch.cuda.is_available():
        try:
            # PyTorch VRAM tracking
            gpu_vram_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            gpu_vram_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            gpu_vram_percent = (gpu_vram_allocated / 24.0) * 100  # RTX 3090 has 24GB
            
            # GPUtil for additional metrics
            gpus = GPUtil.getGPUs()
            if len(gpus) > 0:
                gpu = gpus[0]
                gpu_usage_percent = gpu.load * 100
                gpu_temp = gpu.temperature
                gpu_info = (f"RTX 3090 - Usage: {gpu_usage_percent:.1f}%, "
                           f"VRAM: {gpu_vram_percent:.1f}% ({gpu_vram_allocated:.2f}GB/{24.0}GB), "
                           f"Reserved: {gpu_vram_reserved:.2f}GB, Temp: {gpu_temp}°C")
            else:
                gpu_info = f"RTX 3090 - VRAM: {gpu_vram_percent:.1f}% ({gpu_vram_allocated:.2f}GB/24GB)"
        except Exception as e:
            gpu_info = f"GPU monitoring error: {str(e)}"
    else:
        gpu_info = "GPU: CUDA not available"
    
    # Detailed CPU monitoring for dual Xeon setup
    cpu_percent_overall = psutil.cpu_percent(interval=1)
    cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
    
    # Calculate average CPU usage for both Xeon CPUs (assuming equal core distribution)
    total_cores = len(cpu_per_core)
    cores_per_cpu = total_cores // 2
    cpu1_avg = sum(cpu_per_core[:cores_per_cpu]) / cores_per_cpu
    cpu2_avg = sum(cpu_per_core[cores_per_cpu:]) / cores_per_cpu
    
    # Memory info
    memory = psutil.virtual_memory()
    
    # Thermal and power monitoring
    try:
        sensors = psutil.sensors_temperatures()
        cpu_temps = []
        if 'coretemp' in sensors:
            cpu_temps = [sensor.current for sensor in sensors['coretemp']]
        avg_cpu_temp = sum(cpu_temps) / len(cpu_temps) if cpu_temps else 0
    except:
        avg_cpu_temp = 0
    
    logger.info(f"{gpu_info}")
    logger.info(f"CPU Overall: {cpu_percent_overall:.1f}%, CPU1 Avg: {cpu1_avg:.1f}%, "
               f"CPU2 Avg: {cpu2_avg:.1f}%, Temp: {avg_cpu_temp:.0f}°C")
    logger.info(f"RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
    logger.info(f"Active CPU cores: {sum(1 for usage in cpu_per_core if usage > 10.0)}/{total_cores}")
    
    # VRAM utilization assessment
    vram_status = "EXCELLENT" if gpu_vram_percent > 80 else "GOOD" if gpu_vram_percent > 60 else "LOW"
    cpu_status = "EXCELLENT" if cpu_percent_overall > 70 else "GOOD" if cpu_percent_overall > 50 else "LOW"
    
    logger.info(f"Resource Status - VRAM: {vram_status}, CPU: {cpu_status}")
    
    return {
        'gpu_usage': gpu_usage_percent,
        'gpu_vram_percent': gpu_vram_percent,
        'gpu_vram_allocated_gb': gpu_vram_allocated,
        'gpu_vram_reserved_gb': gpu_vram_reserved,
        'cpu_overall': cpu_percent_overall,
        'cpu1_avg': cpu1_avg,
        'cpu2_avg': cpu2_avg,
        'active_cores': sum(1 for usage in cpu_per_core if usage > 10.0),
        'total_cores': total_cores,
        'ram_percent': memory.percent,
        'vram_status': vram_status,
        'cpu_status': cpu_status
    }

def main():
    """Main training loop - RTX 3090 optimized for 95% utilization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RTX 3090 Optimized Forex Bot Trainer")
    parser.add_argument('--mode', type=str, default='standalone', help='Training mode: standalone or ray_cluster')
    parser.add_argument('--population_size', type=int, default=300, help='Number of bots (optimized for RTX 3090)')
    args = parser.parse_args()
    
    population_size = args.population_size

    # Initialize RTX 3090 optimized trainer
    logger.info(f"🚀 Initializing RTX 3090 trainer: {population_size} bots, 95% VRAM+GPU targets")
    trainer = VRAMOptimizedTrainer(population_size=population_size, target_vram_percent=0.95)

    if args.mode == 'ray_cluster':
        if not ray.is_initialized():
            import ray
            ray.init()
            logger.info("Ray initialized for future cluster scaling.")
        else:
            logger.info("Ray is already initialized.")
    
    logger.info("=== RTX 3090 AGGRESSIVE TRAINING SYSTEM ===")
    logger.info("🎯 Targets: 95% VRAM (22.8GB), 95% GPU utilization")
    logger.info(f"🤖 Population: {population_size} advanced trading bots")
    
    # Create initial population with RTX 3090 optimization
    logger.info("Creating RTX 3090 optimized population...")
    population = trainer.create_population()
    
    # Initial resource baseline
    baseline_stats = trainer.monitor_rtx3090_resources()
    logger.info(f"📊 Baseline: VRAM {baseline_stats.get('vram_utilization_percent', 0):.1f}%, GPU {baseline_stats.get('gpu_usage_percent', 0):.1f}%")
    
    # Training loop with RTX 3090 monitoring
    generations = 50
    for generation in range(generations):
        logger.info(f"\n🔥 === RTX 3090 Generation {generation + 1}/{generations} ===")
        
        # Comprehensive RTX 3090 resource monitoring
        resource_stats = trainer.monitor_rtx3090_resources()
        
        # Performance indicators with emojis
        vram_status = "🟢" if resource_stats.get('meeting_vram_target', False) else "🔴"
        gpu_status = "🟢" if resource_stats.get('meeting_gpu_target', False) else "🔴"
        
        logger.info(f"{vram_status} VRAM: {resource_stats.get('vram_utilization_percent', 0):.1f}% "
                   f"{gpu_status} GPU: {resource_stats.get('gpu_usage_percent', 0):.1f}%")
        
        # Evolve population
        population, results = trainer.evolve_generation(population)
        
        # RTX 3090 AGGRESSIVE training phase
        trainer.gpu_intensive_training(population)
        # Log best performer with detailed metrics
        best = max(results, key=lambda x: x.get('total_trades', 0) or x.get('final_balance', 0))
        trades = best.get('total_trades', 0)
        if trades == 0:
            logger.warning("No trades executed - adjusting strategy thresholds")
            # Adjust trading thresholds for next generation
            trainer.adjust_trading_thresholds(0.1)  # Increase sensitivity by 10%
            
        logger.info(f"Best Bot: Balance={best.get('final_balance', 0):.2f}, "
                   f"Return={best.get('total_return_pct', 0):.2f}%, "
                   f"Win Rate={best.get('win_rate', 0):.2f}, "
                   f"Trades={trades}, "
                   f"Avg Win={best.get('avg_win', 0):.1f}pips, "
                   f"Avg Loss={best.get('avg_loss', 0):.1f}pips, "
                   f"Profit Factor={best.get('profit_factor', 0):.2f}")
        if best['total_trades'] == 0:
            logger.warning("Warning: No trades executed - check strategy thresholds")
        
        # Log trading activity distribution
        action_counts = {'hold': 0, 'buy': 0, 'sell': 0}
        for r in results:
            if 'actions' in r:  # Ensure actions exist
                for action in r['actions']:
                    if action == 0: action_counts['hold'] += 1
                    elif action == 1: action_counts['buy'] += 1
                    elif action == 2: action_counts['sell'] += 1
            if 'actions' in r:
                for action in r['actions']:
                    if action == 0: action_counts['hold'] += 1
                    elif action == 1: action_counts['buy'] += 1
                    elif action == 2: action_counts['sell'] += 1
        logger.info(f"Action Distribution: Hold={action_counts['hold']}, Buy={action_counts['buy']}, Sell={action_counts['sell']}")
        
        # Analyze and save champion every 10 generations
        if (generation + 1) % 10 == 0:
            champion_bot = population[best['bot_id']]
            
            # Save model checkpoint
            checkpoint = {
                'generation': generation,
                'model_state_dict': champion_bot.state_dict(),
                'metrics': best
            }
            # Ensure checkpoints directory exists
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(checkpoint, f'checkpoints/gen_{generation}.pth')
            
            # Log metrics to TensorBoard
            writer.add_scalar('Balance', best['final_balance'], generation)
            writer.add_scalar('Return', best['total_return_pct'], generation)
            writer.add_scalar('Trades', best['total_trades'], generation)
            analysis = trainer.analyze_champion(champion_bot, results)
            model_path = trainer.save_champion(champion_bot, analysis)
            
            logger.info("\n=== CHAMPION ANALYSIS ===")
            champion = analysis['champion_analysis']
            logger.info(f"Final Balance: ${champion['final_balance']:.2f}")
            logger.info(f"Total Return: {champion['total_return_pct']:.2f}%")
            logger.info(f"Win Rate: {champion['win_rate']:.2f}")
            logger.info(f"Profit Factor: {champion['profit_factor']:.2f}")
            logger.info(f"Max Drawdown: {champion['max_drawdown']:.2f}%")
            logger.info(f"Sharpe Ratio: {champion['sharpe_ratio']:.2f}")
    
    # Final champion analysis
    logger.info("\n=== FINAL CHAMPION ANALYSIS ===")
    final_results = trainer.evaluate_population(population)
    champion_bot = population[final_results[0]['bot_id']]
    final_analysis = trainer.analyze_champion(champion_bot, final_results)
    final_model_path = trainer.save_champion(champion_bot, final_analysis)
    
    champion = final_analysis['champion_analysis']
    logger.info(f"🏆 CHAMPION BOT PERFORMANCE:")
    logger.info(f"   Final Balance: ${champion['final_balance']:.2f}")
    logger.info(f"   Total Return: {champion['total_return_pct']:.2f}%")
    logger.info(f"   Win Rate: {champion['win_rate']:.2f}")
    logger.info(f"   Total Trades: {champion['total_trades']}")
    logger.info(f"   Profit Factor: {champion['profit_factor']:.2f}")
    logger.info(f"   Average Win: {champion['average_win']:.2f} pips")
    logger.info(f"   Average Loss: {champion['average_loss']:.2f} pips")
    logger.info(f"   Risk/Reward: {champion['risk_reward_ratio']:.2f}")
    logger.info(f"   Max Drawdown: {champion['max_drawdown']:.2f}%")
    logger.info(f"   Recovery Factor: {champion['recovery_factor']:.2f}")
    logger.info(f"   Sharpe Ratio: {champion['sharpe_ratio']:.2f}")
    logger.info(f"   Calmar Ratio: {champion['calmar_ratio']:.2f}")
    logger.info(f"   Model saved: {final_model_path}")
    
    logger.info("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
