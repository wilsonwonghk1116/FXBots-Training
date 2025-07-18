#!/usr/bin/env python3
"""
Deep Bot Decision Analysis Tool
Study why bots keep choosing HOLD instead of trading
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from run_smart_real_training import SmartTradingBot, SmartForexEnvironment
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BotDecisionAnalyzer:
    """Analyze bot decision-making process in detail"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def analyze_single_bot(self, steps: int = 100):
        """Deep analysis of a single bot's decision process"""
        logger.info("=== DEEP BOT DECISION ANALYSIS ===")
        
        # Create bot and environment
        bot = SmartTradingBot().to(self.device)
        env = SmartForexEnvironment()
        
        logger.info(f"Bot input size: {bot.input_size}")
        logger.info(f"Environment obs space: {env.observation_space.shape}")
        
        # Reset environment
        obs, _ = env.reset()
        logger.info(f"Initial observation shape: {obs.shape}")
        logger.info(f"Initial observation sample: {obs[:5]}")  # First 5 values
        
        # Track decisions
        decisions = []
        rewards_received = []
        action_probs_history = []
        position_sizes_history = []
        confidence_history = []
        
        bot.eval()
        
        for step in range(steps):
            # Get observation and convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Forward pass with detailed analysis
                action_probs, position_size = bot(obs_tensor)
                
                # Extract internal states for analysis
                x_proj = bot.input_projection(obs_tensor)
                x_seq = x_proj.unsqueeze(1)
                lstm1_out, _ = bot.lstm1(x_seq)
                lstm2_out, _ = bot.lstm2(lstm1_out)
                attn_out, _ = bot.attention(lstm2_out, lstm2_out, lstm2_out)
                features = attn_out.squeeze(1)
                
                regime_probs = bot.regime_classifier(features)
                risk_score = bot.risk_network(features)
                confidence = bot.confidence_network(features)
                
                # Log detailed decision process
                if step % 10 == 0:  # Log every 10 steps
                    logger.info(f"\n--- STEP {step} ANALYSIS ---")
                    logger.info(f"Observation range: [{obs.min():.4f}, {obs.max():.4f}]")
                    logger.info(f"Observation std: {obs.std():.4f}")
                    logger.info(f"Action probabilities: {action_probs.cpu().numpy()[0]}")
                    logger.info(f"Position size: {position_size.item():.4f}")
                    logger.info(f"Confidence: {confidence.item():.4f}")
                    logger.info(f"Risk score: {risk_score.item():.4f}")
                    logger.info(f"Regime probs: {regime_probs.cpu().numpy()[0]}")
                    logger.info(f"Current position: {env.position}")
                    logger.info(f"Current balance: {env.balance:.2f}")
                    logger.info(f"Steps since last trade: {step - env.last_trade_step}")
                
                # Select action
                action = torch.argmax(action_probs).item()
                
                # Store for analysis
                decisions.append({
                    'step': step,
                    'action': action,
                    'action_probs': action_probs.cpu().numpy()[0].copy(),
                    'position_size': position_size.item(),
                    'confidence': confidence.item(),
                    'risk_score': risk_score.item(),
                    'regime_probs': regime_probs.cpu().numpy()[0].copy(),
                    'observation': obs.copy(),
                    'env_position': env.position,
                    'env_balance': env.balance,
                    'steps_since_trade': step - env.last_trade_step
                })
                
                action_probs_history.append(action_probs.cpu().numpy()[0])
                position_sizes_history.append(position_size.item())
                confidence_history.append(confidence.item())
            
            # Execute action
            obs, reward, done, _, info = env.step(action, position_size.item())
            rewards_received.append(reward)
            
            if done:
                break
        
        # Analyze results
        self._analyze_decisions(decisions, rewards_received)
        
        return decisions, env
    
    def _analyze_decisions(self, decisions, rewards):
        """Analyze the decision patterns"""
        logger.info("\n=== DECISION ANALYSIS ===")
        
        # Action distribution
        actions = [d['action'] for d in decisions]
        action_counts = {0: actions.count(0), 1: actions.count(1), 2: actions.count(2)}
        total_actions = len(actions)
        
        logger.info(f"Action distribution:")
        logger.info(f"  HOLD (0): {action_counts[0]} ({action_counts[0]/total_actions*100:.1f}%)")
        logger.info(f"  BUY (1):  {action_counts[1]} ({action_counts[1]/total_actions*100:.1f}%)")
        logger.info(f"  SELL (2): {action_counts[2]} ({action_counts[2]/total_actions*100:.1f}%)")
        
        # Analyze action probabilities
        all_probs = np.array([d['action_probs'] for d in decisions])
        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)
        
        logger.info(f"\nMean action probabilities:")
        logger.info(f"  HOLD: {mean_probs[0]:.4f} ± {std_probs[0]:.4f}")
        logger.info(f"  BUY:  {mean_probs[1]:.4f} ± {std_probs[1]:.4f}")
        logger.info(f"  SELL: {mean_probs[2]:.4f} ± {std_probs[2]:.4f}")
        
        # Check for anomalous probabilities
        if np.any(all_probs > 1.0) or np.any(all_probs < 0.0):
            logger.error("🚨 INVALID PROBABILITIES DETECTED!")
            logger.error(f"  Min prob: {all_probs.min()}")
            logger.error(f"  Max prob: {all_probs.max()}")
            logger.error("  This indicates a serious bug in the model's forward pass!")
        
        # Check if probabilities sum to 1
        prob_sums = np.sum(all_probs, axis=1)
        if not np.allclose(prob_sums, 1.0, atol=1e-4):
            logger.error("🚨 PROBABILITIES DON'T SUM TO 1!")
            logger.error(f"  Probability sums range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
            logger.error("  This indicates softmax is not working correctly!")
        
        # Analyze why HOLD is preferred
        if mean_probs[0] > 0.6:  # If HOLD is strongly preferred
            logger.warning("🚨 BOT IS BIASED TOWARD HOLD!")
            logger.info("Possible reasons:")
            
            # Check confidence levels
            confidences = [d['confidence'] for d in decisions]
            mean_confidence = np.mean(confidences)
            logger.info(f"  Mean confidence: {mean_confidence:.4f}")
            if mean_confidence < 0.3:
                logger.info("  ❌ Low confidence causing conservative behavior")
            
            # Check risk scores
            risk_scores = [d['risk_score'] for d in decisions]
            mean_risk = np.mean(risk_scores)
            logger.info(f"  Mean risk score: {mean_risk:.4f}")
            if mean_risk > 0.7:
                logger.info("  ❌ High risk perception causing avoidance")
            
            # Check position sizes
            pos_sizes = [d['position_size'] for d in decisions]
            mean_pos_size = np.mean(pos_sizes)
            logger.info(f"  Mean position size: {mean_pos_size:.4f}")
            if mean_pos_size < 0.1:
                logger.info("  ❌ Very small position sizes indicating fear")
        
        # Analyze observation patterns
        logger.info(f"\nObservation analysis:")
        obs_data = np.array([d['observation'] for d in decisions])
        
        # Price data (first 20 features)
        price_data = obs_data[:, :20]
        logger.info(f"  Price data range: [{price_data.min():.4f}, {price_data.max():.4f}]")
        logger.info(f"  Price data variance: {price_data.var():.6f}")
        
        # Technical indicators (last 6 features)
        tech_data = obs_data[:, 20:]
        logger.info(f"  Technical indicators range: [{tech_data.min():.4f}, {tech_data.max():.4f}]")
        logger.info(f"  Technical indicators variance: {tech_data.var():.6f}")
        
        # Check for NaN or extreme values
        if np.isnan(obs_data).any():
            logger.warning("  ❌ NaN values detected in observations!")
        if np.isinf(obs_data).any():
            logger.warning("  ❌ Infinite values detected in observations!")
        
        # Total rewards
        total_reward = sum(rewards)
        logger.info(f"\nTotal reward received: {total_reward:.2f}")
        if total_reward < -50:  # Mostly holding penalties
            logger.info("  ❌ Mostly holding penalties - bot is being punished for not trading")
    
    def diagnose_model_bias(self):
        """Diagnose potential model architectural biases"""
        logger.info("\n=== MODEL BIAS ANALYSIS ===")
        
        bot = SmartTradingBot().to(self.device)
        bot.eval()
        
        # Test with different input patterns
        test_cases = [
            ("Random input", torch.randn(1, 26, device=self.device)),
            ("Zero input", torch.zeros(1, 26, device=self.device)),
            ("Positive trend", torch.linspace(-1, 1, 26, device=self.device).unsqueeze(0)),
            ("Negative trend", torch.linspace(1, -1, 26, device=self.device).unsqueeze(0)),
            ("High volatility", torch.randn(1, 26, device=self.device) * 2),
        ]
        
        for name, test_input in test_cases:
            with torch.no_grad():
                action_probs, position_size = bot(test_input)
                logger.info(f"{name:15}: HOLD={action_probs[0,0]:.3f}, BUY={action_probs[0,1]:.3f}, SELL={action_probs[0,2]:.3f}, Pos={position_size.item():.3f}")
        
        # Check if model always prefers HOLD
        hold_bias_count = 0
        for name, test_input in test_cases:
            with torch.no_grad():
                action_probs, _ = bot(test_input)
                if action_probs[0, 0] > 0.5:  # HOLD probability > 50%
                    hold_bias_count += 1
        
        if hold_bias_count >= 4:  # Most cases prefer HOLD
            logger.warning("🚨 MODEL HAS STRONG HOLD BIAS!")
            logger.info("Possible fixes:")
            logger.info("1. Increase idle penalty")
            logger.info("2. Reduce confidence weighting")
            logger.info("3. Add exploration noise")
            logger.info("4. Initialize with different weights")

def main():
    """Run comprehensive bot analysis"""
    analyzer = BotDecisionAnalyzer()
    
    # Analyze single bot behavior
    decisions, env = analyzer.analyze_single_bot(steps=200)
    
    # Diagnose model bias
    analyzer.diagnose_model_bias()
    
    # Recommendations
    logger.info("\n=== RECOMMENDATIONS ===")
    
    # Check if no trades occurred
    total_trades = len(env.trades)
    if total_trades == 0:
        logger.warning("🚨 NO TRADES EXECUTED!")
        logger.info("Immediate fixes to try:")
        logger.info("1. Increase holding penalty from -1 to -5")
        logger.info("2. Add exploration noise to action selection")
        logger.info("3. Reduce confidence weighting in action_head")
        logger.info("4. Start with more aggressive position sizes")
        logger.info("5. Reduce trading costs to encourage execution")
    
    # Analyze action probability patterns
    all_probs = np.array([d['action_probs'] for d in decisions])
    hold_dominance = np.mean(all_probs[:, 0])
    
    if hold_dominance > 0.7:
        logger.warning("🚨 SEVERE HOLD BIAS DETECTED!")
        logger.info("Emergency fixes:")
        logger.info("1. Modify confidence weighting in SmartTradingBot.forward()")
        logger.info("2. Change default action distribution from [0.6, 0.2, 0.2] to [0.2, 0.4, 0.4]")
        logger.info("3. Add random exploration with epsilon-greedy strategy")

if __name__ == "__main__":
    main()
