"""
SmartForexEnvironment: Enhanced Forex trading environment for RL bots.
"""

from typing import Optional, Tuple, Dict, Union
import numpy as np
import pandas as pd
import random
import logging
from gymnasium import spaces
import gymnasium as gym
import os
import torch

logger = logging.getLogger(__name__)

class SmartForexEnvironment(gym.Env):
    """Enhanced Forex Environment with difficulty levels"""
    def __init__(self, data_file: str = "data/EURUSD_H1.csv", initial_balance: float = 100000.0):
        super().__init__()
        self.bot_id: Optional[int] = None
        self.total_pnl: float = 0.0
        self.initial_balance = initial_balance
        self.difficulty = 0
        self.data = np.array([])
        self.max_steps = 1000
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
        
        self.data = self._load_data(data_file)
        if len(self.data) == 0:
            self.data = self._generate_synthetic_data()
        self.set_difficulty(self.difficulty)
        self.trading_cost = 0.0002
        self.stop_loss_pips = 30
        self.take_profit_pips = 60
        self.max_position_size = 0.1
        if len(self.data) == 0:
            self.data = self._generate_synthetic_data()
        # Updated observation space to include technical indicators (26 features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def set_difficulty(self, level: int) -> None:
        """Adjust environment difficulty (0=easy, 4=hard)"""
        self.difficulty = max(0, min(4, level))
        self.volatility_multiplier = 1.0 + self.difficulty * 0.25
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = None
        self.trades = []
        self.balance_history = [self.initial_balance]
        self.current_step = 0
        
        # Reset reward/penalty tracking
        self.first_trade_bonus_given = False
        self.last_trade_step = 0
        
        self.trading_cost = 0.0002
        self.stop_loss_pips = 30
        self.take_profit_pips = 60
        self.max_position_size = 0.1
        if len(self.data) == 0:
            self.data = self._generate_synthetic_data()
        # Updated observation space to include technical indicators (26 features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def _load_data(self, data_file: str) -> np.ndarray:
        """Load forex data from CSV file, robust to column name issues"""
        try:
            if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
                df = pd.read_csv(data_file)
                # Normalize column names: strip, lower
                df.columns = [str(col).strip().lower() for col in df.columns]
                # Accept any case for 'close' column
                if 'close' in df.columns:
                    return df['close'].to_numpy().flatten()
                else:
                    logger.warning(f"Could not find 'Close' column in {data_file}. Available columns: {list(df.columns)}")
                    raise ValueError(f"No 'Close' column found in {data_file}. Columns: {list(df.columns)}")
            else:
                return np.array([])
        except Exception as e:
            logger.warning(f"Could not load data from {data_file}: {e}")
            return np.array([])

    def _generate_synthetic_data(self, length: int = 10000) -> np.ndarray:
        """Generate synthetic EUR/USD-like data"""
        logger.info("Generating synthetic forex data for training...")
        np.random.seed(42)
        start_price = 1.1000
        prices = [start_price]
        for i in range(length - 1):
            change = np.random.normal(0, 0.0005)
            trend = 0.000001 * np.sin(i / 100)
            new_price = prices[-1] + change + trend
            new_price = max(0.9000, min(1.3000, new_price))
            prices.append(new_price)
        return np.array(prices)

    def _get_observation(self) -> np.ndarray:
        """Get current market observation with technical indicators (26 features)"""
        # Base price data (20 features)
        if self.current_step < 20:
            price_obs = np.zeros(20)
            available_data = self.data[max(0, self.current_step-19):self.current_step+1]
            price_obs[-len(available_data):] = available_data[-20:]
        else:
            price_obs = self.data[self.current_step-19:self.current_step+1]
        
        if len(price_obs) > 1:
            price_obs = (price_obs - price_obs.mean()) / (price_obs.std() + 1e-8)
        
        # Get technical indicators (6 additional features)
        current_price = self.data[self.current_step] if self.current_step < len(self.data) else 1.1000
        market_analysis = self._intensive_market_analysis(current_price)
        
        # Extract key technical indicators as normalized features
        tech_features = np.zeros(6)
        
        # RSI (normalized to -1 to 1)
        tech_features[0] = (market_analysis.get('rsi', 50) - 50) / 50
        
        # MACD (normalized)
        macd = market_analysis.get('macd', 0)
        tech_features[1] = np.tanh(macd * 10000)  # Normalize MACD
        
        # Bollinger Band position (already 0-1)
        tech_features[2] = market_analysis.get('bb_position', 0.5) * 2 - 1  # Convert to -1 to 1
        
        # Stochastic %K (normalized to -1 to 1)
        tech_features[3] = (market_analysis.get('stoch_k', 50) - 50) / 50
        
        # Momentum 5-period (normalized)
        tech_features[4] = np.tanh(market_analysis.get('momentum_5', 0) / 10)
        
        # Volatility (normalized)
        tech_features[5] = np.tanh(market_analysis.get('volatility', 0.1) / 0.5)
        
        # Combine price data (20) + technical indicators (6) = 26 features
        obs = np.concatenate([price_obs, tech_features])
        return obs.astype(np.float32)

    def step(self, action: Union[int, float], position_size: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment with detailed logging and intensive analysis"""
        # Ensure action is always an int (fix linter error)
        action = int(action)
        if self.entry_price is None:
            self.entry_price = self.data[self.current_step]
        if self.current_step >= len(self.data) - 1:
            logger.info(f"Step {self.current_step}: End of data reached.")
            return self._get_observation(), 0, True, False, {}
        
        current_price = self.data[self.current_step]
        next_price = self.data[self.current_step + 1]
        price_change = next_price - current_price
        
        # INTENSIVE MARKET ANALYSIS (CPU intensive)
        market_analysis = self._intensive_market_analysis(current_price)
        
        # INTENSIVE RISK ASSESSMENT (CPU intensive)
        risk_assessment = self._risk_assessment(market_analysis, action)
        
        # Decision confidence calculation (more CPU work)
        decision_confidence = 0.5
        if 'rsi' in market_analysis and 'trend_risk' in risk_assessment:
            rsi_factor = abs(market_analysis['rsi'] - 50) / 50
            trend_factor = 1.0 - risk_assessment['trend_risk']
            volatility_factor = 1.0 - min(market_analysis.get('volatility', 0.1) / 2.0, 1.0)
            decision_confidence = (rsi_factor + trend_factor + volatility_factor) / 3.0
        
        # Apply analysis results to position sizing (more calculations)
        risk_adjusted_size = position_size * decision_confidence
        if 'volatility_risk' in risk_assessment:
            risk_adjusted_size *= (1.0 - risk_assessment['volatility_risk'] * 0.5)
        
        reward = 0.0
        trade_executed = False
        log_details = {
            'step': self.current_step,
            'action': action,
            'position': self.position,
            'entry_price': self.entry_price,
            'current_price': current_price,
            'next_price': next_price,
            'position_size': position_size,
            'risk_adjusted_size': risk_adjusted_size,
            'decision_confidence': decision_confidence,
            'market_analysis': market_analysis,
            'risk_assessment': risk_assessment
        }
        
        if action == 1 and self.position <= 0:
            direction = "BUY"
            volume = self.max_position_size * risk_adjusted_size
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
                    'step': self.current_step,
                    'decision_confidence': decision_confidence,
                    'risk_metrics': risk_assessment
                })
                logger.info(f"Step {self.current_step}: Closed SHORT at {exit_price}, profit_pips={profit_pips}, trade_pnl_reward={trade_pnl_reward}, confidence={decision_confidence:.2f}")
            
            self.position = 1
            self.entry_price = current_price
            trade_executed = True
            self.last_trade_step = self.current_step  # Update last trade step
            
            # NEW REWARD SYSTEM: First trade bonus
            if not self.first_trade_bonus_given:
                reward += 1000  # First trade bonus
                self.first_trade_bonus_given = True
                logger.info(f"Step {self.current_step}: FIRST TRADE BONUS +1000 reward!")
            
            logger.info(f"Step {self.current_step}: Opened LONG at {current_price}, volume={volume}, confidence={decision_confidence:.2f}")
            
        elif action == 2 and self.position >= 0:
            direction = "SELL"
            volume = self.max_position_size * risk_adjusted_size
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
                    'step': self.current_step,
                    'decision_confidence': decision_confidence,
                    'risk_metrics': risk_assessment
                })
                logger.info(f"Step {self.current_step}: Closed LONG at {current_price}, profit_pips={profit_pips}, trade_pnl_reward={trade_pnl_reward}, confidence={decision_confidence:.2f}")
            
            self.position = -1
            self.entry_price = current_price
            trade_executed = True
            self.last_trade_step = self.current_step  # Update last trade step
            
            # NEW REWARD SYSTEM: First trade bonus
            if not self.first_trade_bonus_given:
                reward += 1000  # First trade bonus
                self.first_trade_bonus_given = True
                logger.info(f"Step {self.current_step}: FIRST TRADE BONUS +1000 reward!")
            
            logger.info(f"Step {self.current_step}: Opened SHORT at {current_price}, volume={volume}, confidence={decision_confidence:.2f}")
            
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
            
        # Only add unrealized PnL if a position is open
        if self.position != 0:
            unrealized_pnl = self.position * (next_price - self.entry_price) * 10000
            # Apply risk-adjusted reward
            risk_multiplier = decision_confidence * (1.0 - risk_assessment.get('volatility_risk', 0.1))
            reward += unrealized_pnl * 0.1 * risk_multiplier
            logger.info(f"Step {self.current_step}: Unrealized PnL={unrealized_pnl}, risk_multiplier={risk_multiplier:.2f}, reward={reward}")
        else:
            logger.info(f"Step {self.current_step}: No open position, reward={reward}")
            
        # Only update balance if a trade was executed or a position is open
        if trade_executed or self.position != 0:
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
            'total_trades': len(self.trades),
            'decision_confidence': decision_confidence,
            'market_analysis': market_analysis,
            'risk_assessment': risk_assessment
        }
        logger.info(f"Step {self.current_step}: Comprehensive analysis complete, confidence={decision_confidence:.2f}")
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
        
        # Reset reward/penalty tracking
        self.first_trade_bonus_given = False
        self.last_trade_step = 0
        
        return self._get_observation(), {}

    def simulate_trading_detailed(self, model, steps: int = 1000) -> Dict:
        """Detailed trading simulation for champion analysis"""
        self.reset()
        total_reward = 0
        device = next(model.parameters()).device
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

    def _intensive_market_analysis(self, current_price: float) -> Dict:
        """Perform intensive market analysis consuming significant CPU resources"""
        analysis = {}
        
        # Technical Indicators (CPU intensive)
        if len(self.data) > 50:
            prices = self.data[max(0, self.current_step-50):self.current_step+1]
            
            # Multiple Moving Averages
            analysis['sma_5'] = np.mean(prices[-5:]) if len(prices) >= 5 else current_price
            analysis['sma_10'] = np.mean(prices[-10:]) if len(prices) >= 10 else current_price
            analysis['sma_20'] = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
            analysis['sma_50'] = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
            
            # Exponential Moving Averages
            alpha_12 = 2 / (12 + 1)
            alpha_26 = 2 / (26 + 1)
            ema_12 = current_price
            ema_26 = current_price
            for i, price in enumerate(prices[-12:]):
                ema_12 = (price * alpha_12) + (ema_12 * (1 - alpha_12))
            for i, price in enumerate(prices[-26:]):
                ema_26 = (price * alpha_26) + (ema_26 * (1 - alpha_26))
            
            analysis['ema_12'] = ema_12
            analysis['ema_26'] = ema_26
            analysis['macd'] = ema_12 - ema_26
            
            # Bollinger Bands
            sma_20 = analysis['sma_20']
            std_20 = np.std(prices[-20:]) if len(prices) >= 20 else 0.001
            analysis['bb_upper'] = sma_20 + (2 * std_20)
            analysis['bb_lower'] = sma_20 - (2 * std_20)
            analysis['bb_position'] = (current_price - analysis['bb_lower']) / (analysis['bb_upper'] - analysis['bb_lower'])
            
            # RSI Calculation (CPU intensive)
            if len(prices) >= 14:
                deltas = np.diff(prices[-15:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / (avg_loss + 1e-8)
                analysis['rsi'] = 100 - (100 / (1 + rs))
            else:
                analysis['rsi'] = 50
            
            # Stochastic Oscillator
            if len(prices) >= 14:
                high_14 = np.max(prices[-14:])
                low_14 = np.min(prices[-14:])
                analysis['stoch_k'] = ((current_price - low_14) / (high_14 - low_14 + 1e-8)) * 100
            else:
                analysis['stoch_k'] = 50
            
            # Price momentum and volatility
            analysis['momentum_5'] = (current_price / prices[-5] - 1) * 100 if len(prices) >= 5 else 0
            analysis['momentum_10'] = (current_price / prices[-10] - 1) * 100 if len(prices) >= 10 else 0
            analysis['volatility'] = np.std(prices[-20:]) * 100 if len(prices) >= 20 else 0.1
            
            # Support and Resistance levels (intensive calculation)
            analysis['support_levels'] = []
            analysis['resistance_levels'] = []
            for i in range(2, min(len(prices)-2, 20)):
                if (prices[-i-1] < prices[-i] > prices[-i+1] and 
                    prices[-i-2] < prices[-i] and prices[-i+2] < prices[-i]):
                    analysis['resistance_levels'].append(prices[-i])
                elif (prices[-i-1] > prices[-i] < prices[-i+1] and 
                      prices[-i-2] > prices[-i] and prices[-i+2] > prices[-i]):
                    analysis['support_levels'].append(prices[-i])
            
            # Fibonacci retracements
            if len(prices) >= 20:
                high = np.max(prices[-20:])
                low = np.min(prices[-20:])
                diff = high - low
                analysis['fib_236'] = high - 0.236 * diff
                analysis['fib_382'] = high - 0.382 * diff
                analysis['fib_500'] = high - 0.500 * diff
                analysis['fib_618'] = high - 0.618 * diff
        
        return analysis
    
    def _risk_assessment(self, analysis: Dict, action: int) -> Dict:
        """Intensive risk assessment consuming more CPU resources"""
        risk_metrics = {}
        
        # Market condition assessment
        if 'rsi' in analysis:
            if analysis['rsi'] > 70:
                risk_metrics['overbought_risk'] = (analysis['rsi'] - 70) / 30
            elif analysis['rsi'] < 30:
                risk_metrics['oversold_risk'] = (30 - analysis['rsi']) / 30
            else:
                risk_metrics['rsi_risk'] = 0.1
        
        # Volatility risk
        if 'volatility' in analysis:
            risk_metrics['volatility_risk'] = min(analysis['volatility'] / 2.0, 1.0)
        
        # Trend strength assessment
        if all(k in analysis for k in ['sma_5', 'sma_10', 'sma_20']):
            if analysis['sma_5'] > analysis['sma_10'] > analysis['sma_20']:
                risk_metrics['trend_strength'] = 'strong_uptrend'
                risk_metrics['trend_risk'] = 0.2 if action == 1 else 0.8
            elif analysis['sma_5'] < analysis['sma_10'] < analysis['sma_20']:
                risk_metrics['trend_strength'] = 'strong_downtrend'  
                risk_metrics['trend_risk'] = 0.2 if action == 2 else 0.8
            else:
                risk_metrics['trend_strength'] = 'sideways'
                risk_metrics['trend_risk'] = 0.6
        
        # Bollinger Band risk
        if 'bb_position' in analysis:
            if analysis['bb_position'] > 0.8:
                risk_metrics['bb_risk'] = 0.8 if action == 1 else 0.3
            elif analysis['bb_position'] < 0.2:
                risk_metrics['bb_risk'] = 0.8 if action == 2 else 0.3
            else:
                risk_metrics['bb_risk'] = 0.3
        
        # Support/Resistance risk
        current_price = self.data[self.current_step]
        if analysis.get('support_levels'):
            closest_support = min(analysis['support_levels'], 
                                key=lambda x: abs(x - current_price))
            risk_metrics['support_distance'] = abs(current_price - closest_support) / current_price
        
        if analysis.get('resistance_levels'):
            closest_resistance = min(analysis['resistance_levels'], 
                                   key=lambda x: abs(x - current_price))
            risk_metrics['resistance_distance'] = abs(current_price - closest_resistance) / current_price
        
        # Monte Carlo simulation for risk (very CPU intensive)
        num_simulations = 100
        future_prices = []
        for _ in range(num_simulations):
            sim_price = current_price
            for step in range(10):
                change = np.random.normal(0, analysis.get('volatility', 0.1) / 100)
                sim_price *= (1 + change)
            future_prices.append(sim_price)
        
        risk_metrics['mc_upside_prob'] = sum(1 for p in future_prices if p > current_price) / num_simulations
        risk_metrics['mc_downside_prob'] = sum(1 for p in future_prices if p < current_price) / num_simulations
        risk_metrics['mc_price_range'] = (min(future_prices), max(future_prices))
        
        return risk_metrics
