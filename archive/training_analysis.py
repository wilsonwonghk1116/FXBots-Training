#!/usr/bin/env python3
"""
MASSIVE SCALE TRAINING RESULTS ANALYZER
=======================================
Analyzes the results from the 200-generation, 200-million-step training
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class MassiveTrainingAnalyzer:
    """Analyzer for massive scale training results"""
    
    def __init__(self, results_dir="training_results"):
        self.results_dir = Path(results_dir)
        self.master_file = self.results_dir / "master_results.json"
        
    def load_master_results(self):
        """Load master results file"""
        if not self.master_file.exists():
            print("❌ Master results file not found")
            return None
        
        with open(self.master_file, 'r') as f:
            return json.load(f)
    
    def analyze_training_progress(self):
        """Analyze overall training progress and performance"""
        master_data = self.load_master_results()
        if not master_data:
            return
        
        print("=" * 80)
        print("MASSIVE SCALE TRAINING ANALYSIS")
        print("=" * 80)
        
        # Basic stats
        total_gens = master_data.get("total_generations", 0)
        completed_gens = len(master_data.get("generations", []))
        episodes_per_gen = master_data.get("episodes_per_generation", 0)
        steps_per_episode = master_data.get("steps_per_episode", 0)
        
        total_episodes = completed_gens * episodes_per_gen
        total_steps = total_episodes * steps_per_episode
        
        print(f"📊 TRAINING SCALE:")
        print(f"   Completed Generations: {completed_gens}/{total_gens}")
        print(f"   Total Episodes: {total_episodes:,}")
        print(f"   Total Trading Steps: {total_steps:,}")
        print(f"   Episodes per Generation: {episodes_per_gen}")
        print(f"   Steps per Episode: {steps_per_episode}")
        
        # Performance analysis
        generations = master_data.get("generations", [])
        if generations:
            self.analyze_performance_metrics(generations)
            self.analyze_learning_progression(generations)
            self.create_performance_charts(generations)
    
    def analyze_performance_metrics(self, generations):
        """Analyze key performance metrics"""
        print(f"\n📈 PERFORMANCE METRICS:")
        
        total_pnl = sum(gen["total_pnl"] for gen in generations)
        total_episodes = sum(gen["total_episodes"] for gen in generations)
        avg_pnl_per_episode = total_pnl / max(1, total_episodes)
        
        print(f"   Total PnL: ${total_pnl:,.2f}")
        print(f"   Average PnL per Episode: ${avg_pnl_per_episode:.2f}")
        
        # Generation-by-generation analysis
        pnls = [gen["total_pnl"] for gen in generations]
        avg_pnls = [gen["avg_pnl_per_episode"] for gen in generations]
        durations = [gen["duration"] for gen in generations]
        
        print(f"   Best Generation PnL: ${max(pnls):,.2f}")
        print(f"   Worst Generation PnL: ${min(pnls):,.2f}")
        print(f"   Average Generation Duration: {np.mean(durations):.1f}s")
        print(f"   Total Training Time: {sum(durations)/3600:.2f} hours")
        
        # Learning trend analysis
        if len(avg_pnls) >= 10:
            recent_avg = np.mean(avg_pnls[-10:])  # Last 10 generations
            early_avg = np.mean(avg_pnls[:10])    # First 10 generations
            improvement = ((recent_avg - early_avg) / abs(early_avg)) * 100
            
            print(f"\n🧠 LEARNING ANALYSIS:")
            print(f"   Early Performance (First 10 gens): ${early_avg:.2f} per episode")
            print(f"   Recent Performance (Last 10 gens): ${recent_avg:.2f} per episode")
            print(f"   Performance Improvement: {improvement:+.2f}%")
            
            if improvement > 5:
                print("   ✅ Strong learning progression detected!")
            elif improvement > 0:
                print("   📈 Positive learning trend")
            else:
                print("   ⚠️ No clear learning improvement")
    
    def analyze_learning_progression(self, generations):
        """Analyze learning progression over time"""
        print(f"\n🧠 REINFORCEMENT LEARNING PROGRESSION:")
        
        # Calculate moving averages
        window_size = min(10, len(generations) // 4)
        avg_pnls = [gen["avg_pnl_per_episode"] for gen in generations]
        
        if len(avg_pnls) >= window_size:
            moving_avg = pd.Series(avg_pnls).rolling(window=window_size).mean()
            
            print(f"   📊 Moving Average Analysis (window={window_size}):")
            print(f"      Starting MA: ${moving_avg.iloc[window_size-1]:.2f}")
            print(f"      Ending MA: ${moving_avg.iloc[-1]:.2f}")
            
            # Trend analysis
            trend_slope = np.polyfit(range(len(moving_avg.dropna())), moving_avg.dropna(), 1)[0]
            print(f"      Learning Trend: ${trend_slope:.4f} per generation")
            
            if trend_slope > 0.1:
                print("      📈 Strong positive learning trend!")
            elif trend_slope > 0:
                print("      📊 Gradual improvement trend")
            else:
                print("      📉 Flat or declining performance")
    
    def create_performance_charts(self, generations):
        """Create performance visualization charts"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare data
            gen_numbers = [gen["generation"] for gen in generations]
            avg_pnls = [gen["avg_pnl_per_episode"] for gen in generations]
            total_pnls = [gen["total_pnl"] for gen in generations]
            durations = [gen["duration"] for gen in generations]
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Massive Scale Training Performance Analysis', fontsize=16)
            
            # Chart 1: Average PnL per Episode
            ax1.plot(gen_numbers, avg_pnls, 'b-', linewidth=2)
            ax1.set_title('Average PnL per Episode by Generation')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Average PnL ($)')
            ax1.grid(True, alpha=0.3)
            
            # Chart 2: Total Generation PnL
            ax2.bar(gen_numbers, total_pnls, alpha=0.7, color='green')
            ax2.set_title('Total PnL by Generation')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Total PnL ($)')
            ax2.grid(True, alpha=0.3)
            
            # Chart 3: Training Duration per Generation
            ax3.plot(gen_numbers, durations, 'r-', linewidth=2)
            ax3.set_title('Training Duration per Generation')
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Duration (seconds)')
            ax3.grid(True, alpha=0.3)
            
            # Chart 4: Cumulative Performance
            cumulative_pnl = np.cumsum(total_pnls)
            ax4.plot(gen_numbers, cumulative_pnl, 'purple', linewidth=3)
            ax4.set_title('Cumulative PnL Over Training')
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Cumulative PnL ($)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.results_dir / "training_analysis_charts.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Performance charts saved to: {chart_path}")
            
            # plt.show()  # Uncomment to display charts
            
        except ImportError:
            print("📊 Chart generation skipped (matplotlib/seaborn not available)")
        except Exception as e:
            print(f"❌ Chart generation failed: {e}")
    
    def generate_detailed_report(self):
        """Generate comprehensive training report"""
        master_data = self.load_master_results()
        if not master_data:
            return
        
        report_file = self.results_dir / "training_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("MASSIVE SCALE FOREX TRADING BOT TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Training configuration
            f.write("TRAINING CONFIGURATION:\n")
            f.write(f"  Total Generations: {master_data.get('total_generations', 0)}\n")
            f.write(f"  Episodes per Generation: {master_data.get('episodes_per_generation', 0)}\n")
            f.write(f"  Steps per Episode: {master_data.get('steps_per_episode', 0)}\n")
            f.write(f"  Training Started: {master_data.get('training_start', 'N/A')}\n\n")
            
            # Performance summary
            generations = master_data.get("generations", [])
            if generations:
                total_pnl = sum(gen["total_pnl"] for gen in generations)
                total_episodes = sum(gen["total_episodes"] for gen in generations)
                
                f.write("PERFORMANCE SUMMARY:\n")
                f.write(f"  Completed Generations: {len(generations)}\n")
                f.write(f"  Total Episodes: {total_episodes:,}\n")
                f.write(f"  Total PnL: ${total_pnl:,.2f}\n")
                f.write(f"  Average PnL per Episode: ${total_pnl/max(1,total_episodes):.2f}\n\n")
                
                # Top performing generations
                sorted_gens = sorted(generations, key=lambda x: x["avg_pnl_per_episode"], reverse=True)
                f.write("TOP 5 PERFORMING GENERATIONS:\n")
                for i, gen in enumerate(sorted_gens[:5]):
                    f.write(f"  {i+1}. Generation {gen['generation']}: ${gen['avg_pnl_per_episode']:.2f} per episode\n")
        
        print(f"📄 Detailed report saved to: {report_file}")

def main():
    """Main analysis function"""
    print("🔍 Starting massive scale training analysis...")
    
    analyzer = MassiveTrainingAnalyzer()
    analyzer.analyze_training_progress()
    analyzer.generate_detailed_report()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
