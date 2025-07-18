#!/usr/bin/env python3
"""
Comprehensive Bot Decision Analysis Tool
========================================

This tool performs deep analysis of why bots keep choosing "hold" actions instead of trading.
It examines:
1. Model architecture and forward pass
2. Action probability distributions
3. Gradient flow and weight initialization
4. Input preprocessing and observation space
5. Reward signal analysis
6. Training stability and convergence
7. Decision confidence and uncertainty
8. Action selection bias and exploration
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
import json
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.model import SmartTradingBot
from src.env import SmartForexEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BotDecisionAnalyzer:
    """Comprehensive analyzer for bot decision-making patterns"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = SmartForexEnvironment()
        self.model = SmartTradingBot().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.info("Using randomly initialized model")
        
        self.analysis_results = {}
        
    def analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze the model architecture for potential issues"""
        logger.info("=== ANALYZING MODEL ARCHITECTURE ===")
        
        results = {
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'layer_info': {},
            'weight_stats': {},
            'gradient_stats': {}
        }
        
        # Analyze each layer
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                results['layer_info'][name] = {
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
                
                # Weight statistics
                for param_name, param in module.named_parameters():
                    full_name = f"{name}.{param_name}"
                    results['weight_stats'][full_name] = {
                        'mean': param.data.mean().item(),
                        'std': param.data.std().item(),
                        'min': param.data.min().item(),
                        'max': param.data.max().item(),
                        'zeros': (param.data == 0).sum().item(),
                        'shape': list(param.shape)
                    }
        
        logger.info(f"Total parameters: {results['total_parameters']:,}")
        logger.info(f"Trainable parameters: {results['trainable_parameters']:,}")
        
        return results
    
    def analyze_forward_pass(self, num_samples: int = 100) -> Dict[str, Any]:
        """Analyze the forward pass for various inputs"""
        logger.info("=== ANALYZING FORWARD PASS ===")
        
        self.model.eval()
        results = {
            'action_distributions': [],
            'position_sizes': [],
            'invalid_probabilities': 0,
            'action_bias': {'hold': 0, 'buy': 0, 'sell': 0},
            'probability_stats': {},
            'activation_patterns': {}
        }
        
        # Test with various inputs
        for i in range(num_samples):
            # Generate random observation
            obs = np.random.randn(26)  # 26 features
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Forward pass
                action_probs, position_size = self.model(obs_tensor)
                
                # Check for invalid probabilities
                action_probs_np = action_probs.cpu().numpy()[0]
                if np.any(np.isnan(action_probs_np)) or np.any(np.isinf(action_probs_np)):
                    results['invalid_probabilities'] += 1
                    logger.warning(f"Invalid probabilities detected: {action_probs_np}")
                
                # Check if probabilities sum to 1
                prob_sum = np.sum(action_probs_np)
                if abs(prob_sum - 1.0) > 1e-5:
                    logger.warning(f"Probabilities don't sum to 1: {action_probs_np}, sum={prob_sum}")
                
                # Record action distributions
                results['action_distributions'].append(action_probs_np.copy())
                results['position_sizes'].append(position_size.cpu().item())
                
                # Count action bias
                action = np.argmax(action_probs_np)
                if action == 0:
                    results['action_bias']['hold'] += 1
                elif action == 1:
                    results['action_bias']['buy'] += 1
                elif action == 2:
                    results['action_bias']['sell'] += 1
        
        # Calculate statistics
        action_probs_array = np.array(results['action_distributions'])
        results['probability_stats'] = {
            'mean_probs': np.mean(action_probs_array, axis=0).tolist(),
            'std_probs': np.std(action_probs_array, axis=0).tolist(),
            'min_probs': np.min(action_probs_array, axis=0).tolist(),
            'max_probs': np.max(action_probs_array, axis=0).tolist()
        }
        
        # Calculate action bias percentages
        total_samples = sum(results['action_bias'].values())
        for action in results['action_bias']:
            results['action_bias'][action] = (results['action_bias'][action] / total_samples) * 100
        
        logger.info(f"Action bias: {results['action_bias']}")
        logger.info(f"Invalid probabilities: {results['invalid_probabilities']}/{num_samples}")
        logger.info(f"Mean action probabilities: {results['probability_stats']['mean_probs']}")
        
        return results
    
    def analyze_softmax_behavior(self) -> Dict[str, Any]:
        """Analyze the softmax layer behavior"""
        logger.info("=== ANALYZING SOFTMAX BEHAVIOR ===")
        
        results = {
            'pre_softmax_values': [],
            'post_softmax_values': [],
            'temperature_analysis': {},
            'saturation_analysis': {}
        }
        
        # Hook to capture pre-softmax values
        pre_softmax_values = []
        
        def hook_fn(module, input, output):
            pre_softmax_values.append(input[0].detach().cpu().numpy())
        
        # Register hook on the action head's final layer (before softmax)
        hook = None
        for name, module in self.model.named_modules():
            if 'action_head' in name and isinstance(module, nn.Linear):
                hook = module.register_forward_hook(hook_fn)
                break
        
        self.model.eval()
        num_samples = 50
        
        for i in range(num_samples):
            obs = np.random.randn(26)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_probs, _ = self.model(obs_tensor)
                results['post_softmax_values'].append(action_probs.cpu().numpy()[0])
        
        if hook:
            hook.remove()
        
        if pre_softmax_values:
            results['pre_softmax_values'] = [v[0] for v in pre_softmax_values]
            
            # Analyze pre-softmax statistics
            pre_softmax_array = np.array(results['pre_softmax_values'])
            results['pre_softmax_stats'] = {
                'mean': np.mean(pre_softmax_array, axis=0).tolist(),
                'std': np.std(pre_softmax_array, axis=0).tolist(),
                'range': (np.max(pre_softmax_array) - np.min(pre_softmax_array)).tolist()
            }
            
            # Check for saturation (very large/small values that would cause softmax to saturate)
            max_vals = np.max(np.abs(pre_softmax_array), axis=1)
            results['saturation_analysis'] = {
                'max_abs_values': max_vals.tolist(),
                'potentially_saturated': np.sum(max_vals > 10).item(),
                'mean_max_abs': np.mean(max_vals).item()
            }
            
            logger.info(f"Pre-softmax stats: {results['pre_softmax_stats']}")
            logger.info(f"Potentially saturated samples: {results['saturation_analysis']['potentially_saturated']}/{num_samples}")
        
        return results
    
    def analyze_input_sensitivity(self) -> Dict[str, Any]:
        """Analyze how sensitive the model is to input changes"""
        logger.info("=== ANALYZING INPUT SENSITIVITY ===")
        
        results = {
            'feature_sensitivity': {},
            'noise_robustness': {},
            'extreme_value_response': {}
        }
        
        self.model.eval()
        
        # Base observation
        base_obs = np.random.randn(26)
        base_obs_tensor = torch.FloatTensor(base_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            base_action_probs, _ = self.model(base_obs_tensor)
            base_action = torch.argmax(base_action_probs).item()
        
        # Test sensitivity to each feature
        for feature_idx in range(26):
            sensitivity_scores = []
            
            for delta in [-2, -1, -0.5, 0.5, 1, 2]:
                modified_obs = base_obs.copy()
                modified_obs[feature_idx] += delta
                modified_obs_tensor = torch.FloatTensor(modified_obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_probs, _ = self.model(modified_obs_tensor)
                    action = torch.argmax(action_probs).item()
                    
                    # Calculate KL divergence between base and modified distributions
                    kl_div = torch.nn.functional.kl_div(
                        torch.log(base_action_probs + 1e-8),
                        action_probs + 1e-8,
                        reduction='sum'
                    ).item()
                    
                    sensitivity_scores.append(kl_div)
            
            results['feature_sensitivity'][f'feature_{feature_idx}'] = {
                'mean_sensitivity': np.mean(sensitivity_scores),
                'max_sensitivity': np.max(sensitivity_scores),
                'sensitivity_scores': sensitivity_scores
            }
        
        # Test noise robustness
        noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
        for noise_level in noise_levels:
            action_changes = 0
            num_tests = 20
            
            for _ in range(num_tests):
                noisy_obs = base_obs + np.random.normal(0, noise_level, 26)
                noisy_obs_tensor = torch.FloatTensor(noisy_obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_probs, _ = self.model(noisy_obs_tensor)
                    action = torch.argmax(action_probs).item()
                    
                    if action != base_action:
                        action_changes += 1
            
            results['noise_robustness'][f'noise_{noise_level}'] = action_changes / num_tests
        
        logger.info("Input sensitivity analysis completed")
        return results
    
    def analyze_training_simulation(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Simulate training episodes and analyze decision patterns"""
        logger.info("=== ANALYZING TRAINING SIMULATION ===")
        
        results = {
            'episode_stats': [],
            'action_sequences': [],
            'reward_patterns': [],
            'decision_confidence': [],
            'trade_execution_analysis': {}
        }
        
        self.model.eval()
        
        for episode in range(num_episodes):
            self.env.reset()
            episode_actions = []
            episode_rewards = []
            episode_confidences = []
            episode_trades = 0
            
            for step in range(100):  # 100 steps per episode
                obs = self.env._get_observation()
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_probs, position_size = self.model(obs_tensor)
                    action = torch.argmax(action_probs).item()
                    
                    # Calculate decision confidence (max probability)
                    confidence = torch.max(action_probs).item()
                    
                    episode_actions.append(action)
                    episode_confidences.append(confidence)
                
                obs, reward, done, _, info = self.env.step(action, position_size.item())
                episode_rewards.append(reward)
                
                if info.get('trade_executed', False):
                    episode_trades += 1
                
                if done:
                    break
            
            # Analyze episode
            action_counts = np.bincount(episode_actions, minlength=3)
            results['episode_stats'].append({
                'episode': episode,
                'total_steps': len(episode_actions),
                'total_reward': sum(episode_rewards),
                'action_counts': action_counts.tolist(),
                'action_percentages': (action_counts / len(episode_actions) * 100).tolist(),
                'trades_executed': episode_trades,
                'avg_confidence': np.mean(episode_confidences),
                'final_balance': self.env.balance
            })
            
            results['action_sequences'].append(episode_actions)
            results['reward_patterns'].append(episode_rewards)
            results['decision_confidence'].append(episode_confidences)
        
        # Aggregate statistics
        all_actions = [action for seq in results['action_sequences'] for action in seq]
        overall_action_counts = np.bincount(all_actions, minlength=3)
        
        results['trade_execution_analysis'] = {
            'overall_action_distribution': (overall_action_counts / len(all_actions) * 100).tolist(),
            'total_episodes_with_trades': sum(1 for ep in results['episode_stats'] if ep['trades_executed'] > 0),
            'avg_trades_per_episode': np.mean([ep['trades_executed'] for ep in results['episode_stats']]),
            'avg_confidence_across_episodes': np.mean([ep['avg_confidence'] for ep in results['episode_stats']])
        }
        
        logger.info(f"Overall action distribution: Hold={results['trade_execution_analysis']['overall_action_distribution'][0]:.1f}%, "
                   f"Buy={results['trade_execution_analysis']['overall_action_distribution'][1]:.1f}%, "
                   f"Sell={results['trade_execution_analysis']['overall_action_distribution'][2]:.1f}%")
        logger.info(f"Episodes with trades: {results['trade_execution_analysis']['total_episodes_with_trades']}/{num_episodes}")
        
        return results
    
    def diagnose_potential_issues(self) -> Dict[str, Any]:
        """Diagnose potential issues causing the hold bias"""
        logger.info("=== DIAGNOSING POTENTIAL ISSUES ===")
        
        issues = {
            'model_issues': [],
            'training_issues': [],
            'environment_issues': [],
            'recommendations': []
        }
        
        # Check model architecture issues
        arch_analysis = self.analysis_results.get('architecture', {})
        
        # Check for weight initialization issues
        weight_stats = arch_analysis.get('weight_stats', {})
        for param_name, stats in weight_stats.items():
            if 'action_head' in param_name:
                if abs(stats['mean']) > 0.1:
                    issues['model_issues'].append(f"Action head {param_name} has large mean ({stats['mean']:.4f}) - may cause bias")
                if stats['std'] < 0.01:
                    issues['model_issues'].append(f"Action head {param_name} has very small std ({stats['std']:.4f}) - may indicate vanishing gradients")
                if stats['std'] > 1.0:
                    issues['model_issues'].append(f"Action head {param_name} has large std ({stats['std']:.4f}) - may cause instability")
        
        # Check forward pass issues
        forward_analysis = self.analysis_results.get('forward_pass', {})
        action_bias = forward_analysis.get('action_bias', {})
        
        if action_bias.get('hold', 0) > 80:
            issues['model_issues'].append(f"Severe hold bias detected ({action_bias['hold']:.1f}%)")
        
        if forward_analysis.get('invalid_probabilities', 0) > 0:
            issues['model_issues'].append(f"Invalid probabilities detected ({forward_analysis['invalid_probabilities']} instances)")
        
        # Check softmax issues
        softmax_analysis = self.analysis_results.get('softmax', {})
        saturation = softmax_analysis.get('saturation_analysis', {})
        
        if saturation.get('potentially_saturated', 0) > 10:
            issues['model_issues'].append(f"Softmax saturation detected ({saturation['potentially_saturated']} instances)")
        
        # Generate recommendations
        if len(issues['model_issues']) > 0:
            issues['recommendations'].extend([
                "Reinitialize model weights with proper initialization (Xavier/He)",
                "Add weight regularization to prevent extreme values",
                "Use gradient clipping during training",
                "Consider reducing learning rate",
                "Add batch normalization before softmax"
            ])
        
        # Check for exploration issues
        training_sim = self.analysis_results.get('training_simulation', {})
        trade_analysis = training_sim.get('trade_execution_analysis', {})
        
        if trade_analysis.get('total_episodes_with_trades', 0) == 0:
            issues['training_issues'].append("No trades executed in any episode - complete exploration failure")
            issues['recommendations'].extend([
                "Implement epsilon-greedy exploration",
                "Add entropy bonus to encourage exploration",
                "Use action noise injection",
                "Implement curiosity-driven exploration"
            ])
        
        logger.info(f"Identified {len(issues['model_issues'])} model issues and {len(issues['training_issues'])} training issues")
        
        return issues
    
    def generate_visualizations(self, save_dir: str = "analysis_plots"):
        """Generate visualization plots for the analysis"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Action distribution plot
        if 'forward_pass' in self.analysis_results:
            action_bias = self.analysis_results['forward_pass']['action_bias']
            
            plt.figure(figsize=(10, 6))
            actions = list(action_bias.keys())
            percentages = list(action_bias.values())
            
            plt.bar(actions, percentages, color=['red', 'green', 'blue'])
            plt.title('Action Selection Bias Distribution')
            plt.xlabel('Actions')
            plt.ylabel('Percentage (%)')
            plt.ylim(0, 100)
            
            for i, v in enumerate(percentages):
                plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'action_bias_distribution.png'), dpi=300)
            plt.close()
        
        # Probability distribution heatmap
        if 'forward_pass' in self.analysis_results:
            action_distributions = self.analysis_results['forward_pass']['action_distributions']
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(np.array(action_distributions), 
                       xticklabels=['Hold', 'Buy', 'Sell'],
                       yticklabels=False,
                       cmap='viridis',
                       cbar_kws={'label': 'Probability'})
            plt.title('Action Probability Distributions Across Samples')
            plt.xlabel('Actions')
            plt.ylabel('Sample Index')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'probability_heatmap.png'), dpi=300)
            plt.close()
        
        logger.info(f"Visualizations saved to {save_dir}")
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run all analysis components"""
        logger.info("Starting comprehensive bot decision analysis...")
        
        # Run all analyses
        self.analysis_results['architecture'] = self.analyze_model_architecture()
        self.analysis_results['forward_pass'] = self.analyze_forward_pass()
        self.analysis_results['softmax'] = self.analyze_softmax_behavior()
        self.analysis_results['input_sensitivity'] = self.analyze_input_sensitivity()
        self.analysis_results['training_simulation'] = self.analyze_training_simulation()
        self.analysis_results['diagnosis'] = self.diagnose_potential_issues()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"bot_decision_analysis_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(self.analysis_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Analysis results saved to {results_file}")
        
        # Print summary
        self.print_analysis_summary()
        
        return self.analysis_results
    
    def print_analysis_summary(self):
        """Print a concise summary of the analysis"""
        print("\n" + "="*80)
        print("BOT DECISION ANALYSIS SUMMARY")
        print("="*80)
        
        # Architecture summary
        arch = self.analysis_results.get('architecture', {})
        print(f"Model Parameters: {arch.get('total_parameters', 'N/A'):,}")
        
        # Forward pass summary
        forward = self.analysis_results.get('forward_pass', {})
        action_bias = forward.get('action_bias', {})
        print(f"Action Bias - Hold: {action_bias.get('hold', 0):.1f}%, "
              f"Buy: {action_bias.get('buy', 0):.1f}%, "
              f"Sell: {action_bias.get('sell', 0):.1f}%")
        print(f"Invalid Probabilities: {forward.get('invalid_probabilities', 0)}")
        
        # Training simulation summary
        training = self.analysis_results.get('training_simulation', {})
        trade_analysis = training.get('trade_execution_analysis', {})
        print(f"Episodes with Trades: {trade_analysis.get('total_episodes_with_trades', 0)}")
        print(f"Avg Trades per Episode: {trade_analysis.get('avg_trades_per_episode', 0):.2f}")
        
        # Issues summary
        diagnosis = self.analysis_results.get('diagnosis', {})
        model_issues = diagnosis.get('model_issues', [])
        training_issues = diagnosis.get('training_issues', [])
        
        print(f"\nISSUES FOUND:")
        print(f"Model Issues: {len(model_issues)}")
        for issue in model_issues[:3]:  # Show first 3
            print(f"  - {issue}")
        
        print(f"Training Issues: {len(training_issues)}")
        for issue in training_issues[:3]:  # Show first 3
            print(f"  - {issue}")
        
        recommendations = diagnosis.get('recommendations', [])
        print(f"\nTOP RECOMMENDATIONS:")
        for rec in recommendations[:5]:  # Show first 5
            print(f"  - {rec}")
        
        print("="*80)

def main():
    """Main function to run the analysis"""
    # First analyze a fresh model to understand base issues
    logger.info("Analyzing fresh randomly initialized model...")
    analyzer = BotDecisionAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    return results

if __name__ == "__main__":
    main()
