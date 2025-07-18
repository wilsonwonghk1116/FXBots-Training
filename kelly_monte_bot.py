#!/usr/bin/env python3
"""
Kelly Monte Carlo Trading Bot System
Implements Monte Carlo simulation with Kelly Criterion for optimal position sizing
Designed for 2000 bot fleet running on 20 years H1 FOREX data
"""

import numpy as np
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Optional numba import for GPU acceleration
try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda x: x  # Dummy decorator
    cuda = None

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingParameters:
    """Trading parameters configuration"""
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    stop_loss_pips: float = 30.0      # Stop loss in pips
    take_profit_pips: float = 60.0    # Take profit in pips
    min_trades_for_update: int = 50   # Minimum trades before updating parameters
    rolling_history_size: int = 1000  # Rolling history size for adaptation
    monte_carlo_scenarios: int = 50000 # Massively increased MC scenarios for GPU saturation
    update_frequency: int = 50        # Update frequency for p and R
    pip_value: float = 0.0001         # Pip value for major pairs

@dataclass
class ScenarioResult:
    """Result of a single Monte Carlo scenario"""
    is_win: bool
    payoff_ratio: float
    entry_price: float
    exit_price: float
    pips_gained: float

@dataclass
class KellyEstimates:
    """Kelly Criterion estimates"""
    win_probability: float
    average_win_payoff: float
    average_loss_payoff: float
    payoff_ratio: float
    kelly_fraction: float
    constrained_fraction: float

class DataManager:
    """Handles historical FOREX data loading and preprocessing"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = Path(data_folder)
        self.price_data = None
        self.returns = None
        
    def load_h1_data(self, currency_pair: str = "EURUSD") -> pd.DataFrame:
        """
        Load 20 years of H1 historical data
        
        Args:
            currency_pair: Currency pair to load (e.g., 'EURUSD')
            
        Returns:
            DataFrame with OHLC data
        """
        try:
            # Try to load from CSV file
            file_path = self.data_folder / f"{currency_pair}_H1.csv"
            if file_path.exists():
                logger.info(f"Loading {currency_pair} data from {file_path}")
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                logger.warning(f"No data file found at {file_path}, generating synthetic data")
                df = self._generate_synthetic_data(currency_pair)
                
            self.price_data = df
            self._preprocess_data()
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self._generate_synthetic_data(currency_pair)
    
    def _generate_synthetic_data(self, currency_pair: str = "EURUSD") -> pd.DataFrame:
        """Generate 20 years of synthetic H1 FOREX data"""
        logger.info(f"Generating 20 years of synthetic {currency_pair} H1 data")
        
        # 20 years * 365 days * 24 hours = 175,200 hours
        n_hours = 20 * 365 * 24
        
        # Start date
        start_date = pd.Timestamp('2004-01-01')
        timestamps = pd.date_range(start=start_date, periods=n_hours, freq='h')
        
        # Generate realistic FOREX price movement
        np.random.seed(42)
        initial_price = 1.2000  # EURUSD starting price
        
        # Parameters for realistic FOREX simulation
        annual_drift = 0.02      # 2% annual drift
        annual_volatility = 0.12 # 12% annual volatility
        
        # Convert to hourly parameters
        dt = 1.0 / (365 * 24)   # Time step in years
        drift = annual_drift * dt
        vol = annual_volatility * np.sqrt(dt)
        
        # Generate price path using geometric Brownian motion
        returns = np.random.normal(drift, vol, n_hours)
        returns[0] = 0  # First return is zero
        
        # Add some regime changes and volatility clustering
        for i in range(1, len(returns)):
            # Volatility clustering (GARCH-like effect)
            vol_factor = 1.0 + 0.1 * abs(returns[i-1]) / vol
            returns[i] *= vol_factor
            
            # Add occasional jumps (news events)
            if np.random.random() < 0.001:  # 0.1% chance of jump
                returns[i] += np.random.normal(0, 0.002) * np.random.choice([-1, 1])
        
        # Convert to price levels
        log_prices = np.cumsum(returns)
        prices = initial_price * np.exp(log_prices)
        
        # Generate OHLC data
        ohlc_data = []
        for i, price in enumerate(prices):
            # Add intraday noise
            noise = np.random.normal(0, 0.0002, 4)  # Small intraday movements
            high = price + abs(noise[0])
            low = price - abs(noise[1])
            open_price = price + noise[2]
            close_price = price + noise[3]
            
            # Ensure OHLC consistency
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            ohlc_data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': np.random.randint(100, 1000)
            })
        
        df = pd.DataFrame(ohlc_data)
        df.set_index('timestamp', inplace=True)
        
        # Save for future use
        self.data_folder.mkdir(exist_ok=True)
        df.to_csv(self.data_folder / f"{currency_pair}_H1.csv")
        
        logger.info(f"Generated {len(df)} hours of synthetic data from {df.index[0]} to {df.index[-1]}")
        return df
    
    def _preprocess_data(self):
        """Preprocess price data into returns series"""
        if self.price_data is None:
            raise ValueError("No price data loaded")
        
        # Calculate returns
        self.returns = self.price_data['close'].pct_change().dropna()
        
        # Calculate additional features
        self.price_data['sma_20'] = self.price_data['close'].rolling(20).mean()
        self.price_data['sma_50'] = self.price_data['close'].rolling(50).mean()
        self.price_data['volatility_20'] = self.returns.rolling(20).std()
        
        logger.info(f"Preprocessed {len(self.returns)} return observations")
    
    def get_return_distribution_params(self, lookback_periods: int = 1000) -> Dict[str, float]:
        """
        Get parameters for return distribution
        
        Args:
            lookback_periods: Number of periods to look back
            
        Returns:
            Dictionary with distribution parameters
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Call _preprocess_data first.")
        
        recent_returns = self.returns.tail(lookback_periods)
        
        return {
            'mean': float(recent_returns.mean()),
            'std': float(recent_returns.std()),
            'skew': float(recent_returns.skew()),
            'kurt': float(recent_returns.kurtosis()),
            'min': float(recent_returns.min()),
            'max': float(recent_returns.max())
        }

class MonteCarloEngine:
    """Monte Carlo simulation engine for scenario generation"""
    
    def __init__(self, params: TradingParameters):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Monte Carlo engine initialized on {self.device}")
    
    def generate_scenarios(self, 
                          current_price: float,
                          return_params: Dict[str, float],
                          entry_signal: str,
                          n_scenarios: int = None) -> List[ScenarioResult]:
        """
        Generate Monte Carlo price path scenarios
        
        Args:
            current_price: Current market price
            return_params: Return distribution parameters
            entry_signal: 'BUY' or 'SELL'
            n_scenarios: Number of scenarios to generate
            
        Returns:
            List of scenario results
        """
        if n_scenarios is None:
            n_scenarios = self.params.monte_carlo_scenarios
        
        # Use GPU acceleration if available
        if self.device.type == 'cuda':
            return self._generate_scenarios_gpu(current_price, return_params, entry_signal, n_scenarios)
        else:
            return self._generate_scenarios_cpu(current_price, return_params, entry_signal, n_scenarios)
    
    def _generate_scenarios_gpu(self, 
                               current_price: float,
                               return_params: Dict[str, float],
                               entry_signal: str,
                               n_scenarios: int) -> List[ScenarioResult]:
        """GPU-accelerated scenario generation with optimized batch processing"""
        # Generate random returns on GPU with larger batches for better utilization
        mean = return_params['mean']
        std = return_params['std']
        
        # Simulate price paths (assume average holding period of 10 hours)
        n_steps = 10
        
        # Process in much larger batches for maximum GPU utilization
        batch_size = min(n_scenarios, 50000)  # Process up to 50k scenarios at once
        scenarios = []
        
        for i in range(0, n_scenarios, batch_size):
            current_batch_size = min(batch_size, n_scenarios - i)
            
            # Generate multiple large random tensors on GPU for maximum utilization
            random_returns = torch.normal(
                mean=mean, 
                std=std, 
                size=(current_batch_size, n_steps),
                device=self.device,
                dtype=torch.float32  # Use float32 for better GPU performance
            )
            
            # Vectorized operations on GPU for maximum efficiency
            price_changes = torch.cumsum(random_returns, dim=1)
            final_price_changes = price_changes[:, -1]
            
            # Vectorized final price calculation
            final_prices = current_price * torch.exp(final_price_changes)
            
            # Keep everything on GPU as long as possible
            final_prices_cpu = final_prices.cpu().numpy()
            
            # Batch process scenario evaluation
            batch_scenarios = self._batch_evaluate_trades(
                current_price, final_prices_cpu, entry_signal
            )
            scenarios.extend(batch_scenarios)
        
        return scenarios
    
    def _generate_scenarios_cpu(self, 
                               current_price: float,
                               return_params: Dict[str, float],
                               entry_signal: str,
                               n_scenarios: int) -> List[ScenarioResult]:
        """CPU-based scenario generation with parallel processing"""
        from concurrent.futures import ProcessPoolExecutor
        
        mean = return_params['mean']
        std = return_params['std']
        
        # Split scenarios across ALL available CPU cores for maximum utilization
        max_workers = mp.cpu_count()  # Use ALL CPU cores for maximum performance
        scenarios_per_worker = max(1000, n_scenarios // max_workers)  # Ensure substantial work per worker
        
        def generate_worker_scenarios(worker_scenarios):
            """Worker function to generate scenarios in parallel"""
            worker_results = []
            for _ in range(worker_scenarios):
                # Simulate price path (assume average holding period of 10 hours)
                n_steps = 10
                returns = np.random.normal(mean, std, n_steps)
                
                # Calculate final price
                cumulative_return = np.sum(returns)
                final_price = current_price * np.exp(cumulative_return)
                
                # Evaluate trade outcome
                scenario = self._evaluate_trade_outcome(
                    current_price, final_price, entry_signal
                )
                worker_results.append(scenario)
            return worker_results
        
        # Use ProcessPoolExecutor for true parallel CPU processing
        all_scenarios = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Distribute scenarios across workers
            worker_loads = []
            remaining = n_scenarios
            for _ in range(max_workers):
                current_load = min(scenarios_per_worker, remaining)
                if current_load > 0:
                    worker_loads.append(current_load)
                    remaining -= current_load
            
            # Submit work to processes
            futures = [executor.submit(generate_worker_scenarios, load) for load in worker_loads]
            
            # Collect results
            for future in futures:
                try:
                    worker_scenarios = future.result()
                    all_scenarios.extend(worker_scenarios)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
        
        return all_scenarios
    
    def _batch_evaluate_trades(self, 
                              entry_price: float,
                              exit_prices: np.ndarray,
                              entry_signal: str) -> List[ScenarioResult]:
        """
        Vectorized batch evaluation of trade outcomes for maximum performance
        
        Args:
            entry_price: Entry price
            exit_prices: Array of exit prices
            entry_signal: 'BUY' or 'SELL'
            
        Returns:
            List of ScenarioResult objects
        """
        # Vectorized calculations for all scenarios at once
        if entry_signal.upper() == 'BUY':
            pips_gained = (exit_prices - entry_price) / self.params.pip_value
            is_win = exit_prices > entry_price
        else:  # SELL
            pips_gained = (entry_price - exit_prices) / self.params.pip_value
            is_win = exit_prices < entry_price
        
        # Vectorized stop loss and take profit logic
        abs_pips = np.abs(pips_gained)
        
        # Apply take profit
        tp_mask = (abs_pips >= self.params.take_profit_pips) & is_win
        pips_gained[tp_mask] = np.where(
            entry_signal.upper() == 'BUY',
            self.params.take_profit_pips,
            self.params.take_profit_pips
        )
        
        # Apply stop loss
        sl_mask = (abs_pips >= self.params.stop_loss_pips) & ~is_win
        pips_gained[sl_mask] = -self.params.stop_loss_pips
        
        # Calculate payoff ratios vectorized
        payoff_ratios = np.where(
            pips_gained < 0,
            pips_gained / self.params.stop_loss_pips,
            pips_gained / self.params.stop_loss_pips
        )
        
        # Update win status after TP/SL application
        final_is_win = pips_gained > 0
        
        # Convert to ScenarioResult objects
        scenarios = []
        for i in range(len(exit_prices)):
            scenarios.append(ScenarioResult(
                is_win=bool(final_is_win[i]),
                payoff_ratio=float(payoff_ratios[i]),
                entry_price=entry_price,
                exit_price=float(exit_prices[i]),
                pips_gained=float(pips_gained[i])
            ))
        
        return scenarios
    
    def _evaluate_trade_outcome(self, 
                               entry_price: float,
                               exit_price: float,
                               entry_signal: str) -> ScenarioResult:
        """
        Evaluate the outcome of a simulated trade
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            entry_signal: 'BUY' or 'SELL'
            
        Returns:
            ScenarioResult with trade outcome
        """
        if entry_signal.upper() == 'BUY':
            pips_gained = (exit_price - entry_price) / self.params.pip_value
            is_win = exit_price > entry_price
        else:  # SELL
            pips_gained = (entry_price - exit_price) / self.params.pip_value
            is_win = exit_price < entry_price
        
        # Apply stop loss and take profit logic
        abs_pips = abs(pips_gained)
        
        if abs_pips >= self.params.take_profit_pips and is_win:
            # Hit take profit
            pips_gained = self.params.take_profit_pips if entry_signal.upper() == 'BUY' else self.params.take_profit_pips
            payoff_ratio = self.params.take_profit_pips / self.params.stop_loss_pips
            is_win = True
        elif abs_pips >= self.params.stop_loss_pips and not is_win:
            # Hit stop loss
            pips_gained = -self.params.stop_loss_pips
            payoff_ratio = -1.0
            is_win = False
        else:
            # Natural exit
            payoff_ratio = pips_gained / self.params.stop_loss_pips if pips_gained < 0 else pips_gained / self.params.stop_loss_pips
        
        return ScenarioResult(
            is_win=bool(is_win),
            payoff_ratio=float(payoff_ratio),
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            pips_gained=float(pips_gained)
        )

class KellyCalculator:
    """Kelly Criterion calculator for position sizing"""
    
    def __init__(self, params: TradingParameters):
        self.params = params
    
    def estimate_parameters(self, scenarios: List[ScenarioResult]) -> KellyEstimates:
        """
        Estimate win probability and payoff ratios from Monte Carlo scenarios
        
        Args:
            scenarios: List of scenario results
            
        Returns:
            KellyEstimates with calculated parameters
        """
        if not scenarios:
            raise ValueError("No scenarios provided")
        
        # Calculate win probability
        wins = [s for s in scenarios if s.is_win]
        losses = [s for s in scenarios if not s.is_win]
        
        win_probability = len(wins) / len(scenarios)
        
        # Calculate average payoffs
        if wins:
            avg_win_payoff = np.mean([s.payoff_ratio for s in wins])
        else:
            avg_win_payoff = 0.0
        
        if losses:
            avg_loss_payoff = abs(np.mean([s.payoff_ratio for s in losses]))
        else:
            avg_loss_payoff = 1.0  # Default to 1:1 if no losses
        
        # Calculate overall payoff ratio (R)
        payoff_ratio = avg_win_payoff / avg_loss_payoff if avg_loss_payoff > 0 else 0.0
        
        # Calculate Kelly fraction: f* = p - (1-p)/R
        if payoff_ratio > 0:
            kelly_fraction = win_probability - (1 - win_probability) / payoff_ratio
        else:
            kelly_fraction = 0.0
        
        # Apply constraints
        constrained_fraction = self._apply_constraints(kelly_fraction)
        
        return KellyEstimates(
            win_probability=win_probability,
            average_win_payoff=avg_win_payoff,
            average_loss_payoff=avg_loss_payoff,
            payoff_ratio=payoff_ratio,
            kelly_fraction=kelly_fraction,
            constrained_fraction=constrained_fraction
        )
    
    def _apply_constraints(self, kelly_fraction: float) -> float:
        """
        Apply practical constraints to Kelly fraction
        
        Args:
            kelly_fraction: Raw Kelly fraction
            
        Returns:
            Constrained Kelly fraction
        """
        # Floor at zero (no negative positions)
        constrained = max(0.0, kelly_fraction)
        
        # Cap at maximum risk per trade
        constrained = min(constrained, self.params.max_risk_per_trade)
        
        # Additional practical constraints
        # Kelly can be overly aggressive, so we often use a fraction of Kelly
        kelly_multiplier = 0.25  # Use 25% of Kelly for safety
        constrained *= kelly_multiplier
        
        return constrained
    
    def calculate_position_size(self, 
                               kelly_estimates: KellyEstimates,
                               current_equity: float) -> float:
        """
        Calculate position size based on Kelly fraction and current equity
        
        Args:
            kelly_estimates: Kelly estimates
            current_equity: Current account equity
            
        Returns:
            Position size in base currency units
        """
        position_size = kelly_estimates.constrained_fraction * current_equity
        return position_size

class KellyMonteBot:
    """
    Main Kelly Monte Carlo Trading Bot implementation
    Combines Monte Carlo simulation with Kelly Criterion for optimal position sizing
    """
    
    def __init__(self, 
                 bot_id: int,
                 initial_equity: float = 100000.0,
                 params: TradingParameters = None):
        self.bot_id = bot_id
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.params = params or TradingParameters()
        
        # Initialize components
        self.data_manager = DataManager()
        self.monte_carlo = MonteCarloEngine(self.params)
        self.kelly_calculator = KellyCalculator(self.params)
        
        # Trading state
        self.trade_history: List[Dict] = []
        self.current_position = None
        self.returns_params = None
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pips = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_equity
        
        logger.info(f"KellyMonteBot {bot_id} initialized with {initial_equity} equity")
    
    def initialize(self, currency_pair: str = "EURUSD"):
        """
        Initialize the bot with historical data
        
        Args:
            currency_pair: Currency pair to trade
        """
        logger.info(f"Bot {self.bot_id}: Initializing with {currency_pair} data")
        
        # Load and preprocess data
        self.data_manager.load_h1_data(currency_pair)
        
        # Get initial return distribution parameters
        self.returns_params = self.data_manager.get_return_distribution_params()
        
        logger.info(f"Bot {self.bot_id}: Initialized with return params: {self.returns_params}")
    
    def generate_trading_signal(self, current_data: pd.Series) -> Optional[str]:
        """
        Generate trading signal based on market conditions
        Simple momentum strategy for demonstration
        
        Args:
            current_data: Current market data row
            
        Returns:
            'BUY', 'SELL', or None
        """
        if pd.isna(current_data.get('sma_20')) or pd.isna(current_data.get('sma_50')):
            return None
        
        # Simple moving average crossover strategy
        if current_data['close'] > current_data['sma_20'] > current_data['sma_50']:
            return 'BUY'
        elif current_data['close'] < current_data['sma_20'] < current_data['sma_50']:
            return 'SELL'
        
        return None
    
    def make_trading_decision(self, 
                            current_price: float,
                            market_data: pd.Series,
                            timestamp: pd.Timestamp) -> Optional[Dict]:
        """
        Make trading decision using Monte Carlo simulation and Kelly Criterion
        
        Args:
            current_price: Current market price
            market_data: Current market data
            timestamp: Current timestamp
            
        Returns:
            Trading decision dictionary or None if no trade
        """
        # Generate trading signal
        signal = self.generate_trading_signal(market_data)
        if signal is None:
            return None
        
        # Generate Monte Carlo scenarios
        start_time = time.time()
        scenarios = self.monte_carlo.generate_scenarios(
            current_price=current_price,
            return_params=self.returns_params,
            entry_signal=signal,
            n_scenarios=self.params.monte_carlo_scenarios
        )
        mc_time = time.time() - start_time
        
        # Calculate Kelly estimates
        kelly_estimates = self.kelly_calculator.estimate_parameters(scenarios)
        
        # Calculate position size
        position_size = self.kelly_calculator.calculate_position_size(
            kelly_estimates, self.current_equity
        )
        
        # Only trade if Kelly fraction is positive and significant
        if kelly_estimates.constrained_fraction < 0.001:  # Less than 0.1%
            return None
        
        decision = {
            'timestamp': timestamp,
            'signal': signal,
            'entry_price': current_price,
            'position_size': position_size,
            'kelly_estimates': kelly_estimates,
            'mc_scenarios': len(scenarios),
            'mc_computation_time': mc_time,
            'equity_before': self.current_equity
        }
        
        return decision
    
    def execute_trade(self, decision: Dict) -> Dict:
        """
        Execute trade based on decision
        
        Args:
            decision: Trading decision from make_trading_decision
            
        Returns:
            Trade execution result
        """
        # Simulate trade execution
        entry_price = decision['entry_price']
        signal = decision['signal']
        position_size = decision['position_size']
        
        # Calculate stop loss and take profit levels
        if signal == 'BUY':
            stop_loss = entry_price - (self.params.stop_loss_pips * self.params.pip_value)
            take_profit = entry_price + (self.params.take_profit_pips * self.params.pip_value)
        else:  # SELL
            stop_loss = entry_price + (self.params.stop_loss_pips * self.params.pip_value)
            take_profit = entry_price - (self.params.take_profit_pips * self.params.pip_value)
        
        trade_result = {
            'trade_id': len(self.trade_history) + 1,
            'bot_id': self.bot_id,
            'timestamp': decision['timestamp'],
            'signal': signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'kelly_fraction': decision['kelly_estimates'].constrained_fraction,
            'win_probability': decision['kelly_estimates'].win_probability,
            'payoff_ratio': decision['kelly_estimates'].payoff_ratio,
            'status': 'OPEN'
        }
        
        # Update current position
        self.current_position = trade_result
        self.total_trades += 1
        
        return trade_result
    
    def close_trade(self, exit_price: float, exit_reason: str) -> Dict:
        """
        Close current trade and update statistics
        
        Args:
            exit_price: Exit price
            exit_reason: Reason for exit ('STOP_LOSS', 'TAKE_PROFIT', 'NATURAL')
            
        Returns:
            Closed trade result
        """
        if self.current_position is None:
            raise ValueError("No open position to close")
        
        trade = self.current_position.copy()
        trade['exit_price'] = exit_price
        trade['exit_reason'] = exit_reason
        trade['status'] = 'CLOSED'
        
        # Calculate profit/loss
        if trade['signal'] == 'BUY':
            pips_gained = (exit_price - trade['entry_price']) / self.params.pip_value
        else:  # SELL
            pips_gained = (trade['entry_price'] - exit_price) / self.params.pip_value
        
        trade['pips_gained'] = pips_gained
        
        # Calculate P&L in currency terms
        pip_value_currency = 10.0  # $10 per pip for standard lot
        position_lots = trade['position_size'] / 100000  # Convert to lots
        trade_pnl = pips_gained * pip_value_currency * position_lots
        
        trade['pnl'] = trade_pnl
        
        # Update equity
        self.current_equity += trade_pnl
        trade['equity_after'] = self.current_equity
        
        # Update statistics
        if pips_gained > 0:
            self.winning_trades += 1
        
        self.total_pips += pips_gained
        
        # Update drawdown
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Add to trade history
        self.trade_history.append(trade)
        self.current_position = None
        
        # Update return parameters if enough trades
        if len(self.trade_history) % self.params.update_frequency == 0:
            self._update_return_parameters()
        
        return trade
    
    def _update_return_parameters(self):
        """Update return distribution parameters based on recent market conditions"""
        # Get updated parameters from recent data
        self.returns_params = self.data_manager.get_return_distribution_params(
            lookback_periods=min(1000, len(self.data_manager.returns))
        )
        
        logger.info(f"Bot {self.bot_id}: Updated return parameters after {len(self.trade_history)} trades")
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trade_history:
            return {}
        
        closed_trades = [t for t in self.trade_history if t['status'] == 'CLOSED']
        
        if not closed_trades:
            return {}
        
        # Basic metrics
        total_pnl = sum(t['pnl'] for t in closed_trades)
        win_rate = self.winning_trades / len(closed_trades) if closed_trades else 0
        
        # Calculate returns series for Sharpe ratio
        equity_series = [t['equity_after'] for t in closed_trades]
        if len(equity_series) > 1:
            returns = pd.Series(equity_series).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Average metrics
        avg_win = np.mean([t['pnl'] for t in closed_trades if t['pnl'] > 0]) if self.winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in closed_trades if t['pnl'] < 0]) if (len(closed_trades) - self.winning_trades) > 0 else 0
        
        profit_factor = abs(avg_win * self.winning_trades / (avg_loss * (len(closed_trades) - self.winning_trades))) if avg_loss != 0 else float('inf')
        
        return {
            'bot_id': self.bot_id,
            'total_trades': len(closed_trades),
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': (self.current_equity - self.initial_equity) / self.initial_equity * 100,
            'current_equity': self.current_equity,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'total_pips': self.total_pips,
            'trade_history': self.trade_history
        }

class BotFleetManager:
    """
    Manages fleet of 2000 Kelly Monte Carlo bots
    Handles parallel execution and monitoring
    """
    
    def __init__(self, 
                 n_bots: int = 2000,
                 initial_equity: float = 100000.0,
                 params: TradingParameters = None):
        self.n_bots = n_bots
        self.initial_equity = initial_equity
        self.params = params or TradingParameters()
        
        # Initialize bot fleet
        self.bots: List[KellyMonteBot] = []
        self._initialize_fleet()
        
        # Performance tracking
        self.fleet_metrics = {}
        
        logger.info(f"BotFleetManager initialized with {n_bots} bots")
    
    def _initialize_fleet(self):
        """Initialize fleet of trading bots"""
        logger.info(f"Initializing fleet of {self.n_bots} bots...")
        
        for i in range(self.n_bots):
            bot = KellyMonteBot(
                bot_id=i,
                initial_equity=self.initial_equity,
                params=self.params
            )
            self.bots.append(bot)
        
        # Initialize all bots with data (parallel) - Use ALL available cores for maximum performance
        max_workers = mp.cpu_count()  # Use ALL CPU cores
        logger.info(f"Using {max_workers} workers for bot initialization (ALL {mp.cpu_count()} cores)")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(bot.initialize, "EURUSD") for bot in self.bots]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Bot initialization failed: {e}")
        
        logger.info("Fleet initialization complete")
    
    def run_parallel_decisions(self, 
                              current_price: float,
                              market_data: pd.Series,
                              timestamp: pd.Timestamp) -> List[Optional[Dict]]:
        """
        Run trading decisions for all bots in parallel
        
        Args:
            current_price: Current market price
            market_data: Market data for current timestamp
            timestamp: Current timestamp
            
        Returns:
            List of trading decisions (None for no decision)
        """
        decisions = []
        
        # Use parallel processing for decision making - Use ALL available cores for maximum performance
        max_workers = mp.cpu_count()  # Use ALL CPU cores
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    bot.make_trading_decision,
                    current_price,
                    market_data,
                    timestamp
                ) for bot in self.bots
            ]
            
            for future in as_completed(futures):
                try:
                    decision = future.result()
                    decisions.append(decision)
                except Exception as e:
                    logger.error(f"Bot decision failed: {e}")
                    decisions.append(None)
        
        return decisions
    
    def get_fleet_performance(self) -> Dict:
        """
        Get aggregated fleet performance metrics
        
        Returns:
            Fleet performance dictionary
        """
        bot_metrics = []
        
        # Collect metrics from all bots - Use ALL available cores for maximum performance
        max_workers = mp.cpu_count()  # Use ALL CPU cores
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(bot.get_performance_metrics) for bot in self.bots]
            
            for future in as_completed(futures):
                try:
                    metrics = future.result()
                    if metrics:  # Only include bots with trades
                        bot_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Failed to get bot metrics: {e}")
        
        if not bot_metrics:
            return {}
        
        # Aggregate metrics
        total_trades = sum(m['total_trades'] for m in bot_metrics)
        total_winning = sum(m['winning_trades'] for m in bot_metrics)
        total_pnl = sum(m['total_pnl'] for m in bot_metrics)
        total_equity = sum(m['current_equity'] for m in bot_metrics)
        
        avg_return = np.mean([m['total_return_pct'] for m in bot_metrics])
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in bot_metrics if not np.isnan(m['sharpe_ratio'])])
        avg_drawdown = np.mean([m['max_drawdown'] for m in bot_metrics])
        
        fleet_metrics = {
            'n_active_bots': len(bot_metrics),
            'total_trades': total_trades,
            'total_winning_trades': total_winning,
            'fleet_win_rate': total_winning / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'total_equity': total_equity,
            'average_return_pct': avg_return,
            'average_sharpe_ratio': avg_sharpe,
            'average_max_drawdown': avg_drawdown,
            'best_performer': max(bot_metrics, key=lambda x: x['total_return_pct']),
            'worst_performer': min(bot_metrics, key=lambda x: x['total_return_pct'])
        }
        
        return fleet_metrics
    
    def save_results(self, filename: str):
        """Save fleet results to JSON file"""
        results = {
            'fleet_performance': self.get_fleet_performance(),
            'bot_metrics': [bot.get_performance_metrics() for bot in self.bots],
            'parameters': {
                'n_bots': self.n_bots,
                'initial_equity': self.initial_equity,
                'trading_params': self.params.__dict__
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Fleet results saved to {filename}")

if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing Kelly Monte Carlo Bot System")
    
    # Test single bot
    bot = KellyMonteBot(bot_id=0, initial_equity=100000.0)
    bot.initialize("EURUSD")
    
    # Test decision making
    current_price = 1.2000
    market_data = pd.Series({
        'close': current_price,
        'sma_20': 1.1980,
        'sma_50': 1.1950,
        'volume': 1000
    })
    
    decision = bot.make_trading_decision(
        current_price=current_price,
        market_data=market_data,
        timestamp=pd.Timestamp.now()
    )
    
    if decision:
        logger.info(f"Trading decision: {decision['signal']} with Kelly fraction: {decision['kelly_estimates'].constrained_fraction:.4f}")
        
        # Execute trade
        trade_result = bot.execute_trade(decision)
        logger.info(f"Trade executed: {trade_result}")
        
        # Simulate closing the trade
        exit_price = current_price + 0.0020  # 20 pips profit
        closed_trade = bot.close_trade(exit_price, "TAKE_PROFIT")
        logger.info(f"Trade closed: P&L = {closed_trade['pnl']:.2f}")
        
        # Get performance metrics
        metrics = bot.get_performance_metrics()
        logger.info(f"Bot performance: {metrics}")
    
    logger.info("Kelly Monte Carlo Bot System test completed")
