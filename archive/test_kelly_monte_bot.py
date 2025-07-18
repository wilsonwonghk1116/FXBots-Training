#!/usr/bin/env python3
"""
Unit Tests for Kelly Monte Carlo Bot System
Tests Kelly fraction calculation, Monte Carlo scenarios, and parameter estimation
"""

import unittest
import numpy as np
import pandas as pd
import torch
from kelly_monte_bot import (
    KellyMonteBot, 
    KellyCalculator, 
    MonteCarloEngine,
    TradingParameters,
    ScenarioResult,
    KellyEstimates,
    DataManager
)

class TestKellyCalculator(unittest.TestCase):
    """Test Kelly Criterion calculations"""
    
    def setUp(self):
        self.params = TradingParameters()
        self.kelly_calc = KellyCalculator(self.params)
    
    def test_kelly_formula_basic(self):
        """Test basic Kelly formula calculation"""
        # Create simple scenarios: 60% win rate, 2:1 payoff ratio
        scenarios = []
        
        # 60 winning scenarios
        for _ in range(60):
            scenarios.append(ScenarioResult(
                is_win=True,
                payoff_ratio=2.0,
                entry_price=1.2000,
                exit_price=1.2020,
                pips_gained=20.0
            ))
        
        # 40 losing scenarios
        for _ in range(40):
            scenarios.append(ScenarioResult(
                is_win=False,
                payoff_ratio=-1.0,
                entry_price=1.2000,
                exit_price=1.1990,
                pips_gained=-10.0
            ))
        
        estimates = self.kelly_calc.estimate_parameters(scenarios)
        
        # Test calculations
        self.assertAlmostEqual(estimates.win_probability, 0.6, places=2)
        self.assertAlmostEqual(estimates.payoff_ratio, 2.0, places=1)
        
        # Kelly formula: f* = p - (1-p)/R = 0.6 - 0.4/2 = 0.4
        expected_kelly = 0.6 - 0.4/2.0
        self.assertAlmostEqual(estimates.kelly_fraction, expected_kelly, places=2)
    
    def test_kelly_constraints(self):
        """Test Kelly fraction constraints"""
        # Create scenario with very high Kelly fraction
        scenarios = []
        
        # 90% win rate, 10:1 payoff ratio (unrealistic but tests constraints)
        for _ in range(90):
            scenarios.append(ScenarioResult(
                is_win=True,
                payoff_ratio=10.0,
                entry_price=1.2000,
                exit_price=1.2100,
                pips_gained=100.0
            ))
        
        for _ in range(10):
            scenarios.append(ScenarioResult(
                is_win=False,
                payoff_ratio=-1.0,
                entry_price=1.2000,
                exit_price=1.1990,
                pips_gained=-10.0
            ))
        
        estimates = self.kelly_calc.estimate_parameters(scenarios)
        
        # Raw Kelly should be very high
        self.assertGreater(estimates.kelly_fraction, 0.5)
        
        # But constrained Kelly should be <= max_risk_per_trade
        self.assertLessEqual(estimates.constrained_fraction, self.params.max_risk_per_trade)
        self.assertGreaterEqual(estimates.constrained_fraction, 0.0)
    
    def test_negative_kelly(self):
        """Test negative Kelly fraction handling"""
        # Create losing scenarios
        scenarios = []
        
        # 30% win rate, 1:1 payoff ratio -> negative Kelly
        for _ in range(30):
            scenarios.append(ScenarioResult(
                is_win=True,
                payoff_ratio=1.0,
                entry_price=1.2000,
                exit_price=1.2010,
                pips_gained=10.0
            ))
        
        for _ in range(70):
            scenarios.append(ScenarioResult(
                is_win=False,
                payoff_ratio=-1.0,
                entry_price=1.2000,
                exit_price=1.1990,
                pips_gained=-10.0
            ))
        
        estimates = self.kelly_calc.estimate_parameters(scenarios)
        
        # Kelly formula: f* = 0.3 - 0.7/1 = -0.4 (negative)
        expected_kelly = 0.3 - 0.7/1.0
        self.assertAlmostEqual(estimates.kelly_fraction, expected_kelly, places=2)
        
        # Constrained Kelly should be 0 (no negative positions)
        self.assertEqual(estimates.constrained_fraction, 0.0)
    
    def test_position_sizing(self):
        """Test position size calculation"""
        estimates = KellyEstimates(
            win_probability=0.6,
            average_win_payoff=2.0,
            average_loss_payoff=1.0,
            payoff_ratio=2.0,
            kelly_fraction=0.4,
            constrained_fraction=0.01  # 1% of equity
        )
        
        current_equity = 100000.0
        position_size = self.kelly_calc.calculate_position_size(estimates, current_equity)
        
        expected_size = 0.01 * 100000.0  # 1% of 100k
        self.assertEqual(position_size, expected_size)

class TestMonteCarloEngine(unittest.TestCase):
    """Test Monte Carlo scenario generation"""
    
    def setUp(self):
        self.params = TradingParameters(monte_carlo_scenarios=100)
        self.mc_engine = MonteCarloEngine(self.params)
    
    def test_scenario_generation_cpu(self):
        """Test CPU-based scenario generation"""
        current_price = 1.2000
        return_params = {
            'mean': 0.0001,
            'std': 0.001,
            'skew': 0.0,
            'kurt': 0.0,
            'min': -0.01,
            'max': 0.01
        }
        
        scenarios = self.mc_engine._generate_scenarios_cpu(
            current_price, return_params, 'BUY', 100
        )
        
        self.assertEqual(len(scenarios), 100)
        
        for scenario in scenarios:
            self.assertIsInstance(scenario, ScenarioResult)
            self.assertIsInstance(scenario.is_win, bool)
            self.assertIsInstance(scenario.payoff_ratio, float)
            self.assertGreater(scenario.entry_price, 0)
            self.assertGreater(scenario.exit_price, 0)
    
    def test_scenario_generation_gpu(self):
        """Test GPU-based scenario generation if available"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        current_price = 1.2000
        return_params = {
            'mean': 0.0001,
            'std': 0.001,
            'skew': 0.0,
            'kurt': 0.0,
            'min': -0.01,
            'max': 0.01
        }
        
        scenarios = self.mc_engine._generate_scenarios_gpu(
            current_price, return_params, 'BUY', 100
        )
        
        self.assertEqual(len(scenarios), 100)
        
        for scenario in scenarios:
            self.assertIsInstance(scenario, ScenarioResult)
            self.assertIsInstance(scenario.is_win, bool)
            self.assertIsInstance(scenario.payoff_ratio, float)
    
    def test_trade_outcome_evaluation_buy(self):
        """Test trade outcome evaluation for BUY signal"""
        entry_price = 1.2000
        
        # Test winning BUY trade
        exit_price = 1.2060  # +60 pips, should hit take profit
        result = self.mc_engine._evaluate_trade_outcome(entry_price, exit_price, 'BUY')
        
        self.assertTrue(result.is_win)
        self.assertEqual(result.pips_gained, self.params.take_profit_pips)
        
        # Test losing BUY trade
        exit_price = 1.1970  # -30 pips, should hit stop loss
        result = self.mc_engine._evaluate_trade_outcome(entry_price, exit_price, 'BUY')
        
        self.assertFalse(result.is_win)
        self.assertAlmostEqual(result.pips_gained, -self.params.stop_loss_pips, places=1)
    
    def test_trade_outcome_evaluation_sell(self):
        """Test trade outcome evaluation for SELL signal"""
        entry_price = 1.2000
        
        # Test winning SELL trade
        exit_price = 1.1940  # -60 pips for price, +60 pips for SELL
        result = self.mc_engine._evaluate_trade_outcome(entry_price, exit_price, 'SELL')
        
        self.assertTrue(result.is_win)
        self.assertEqual(result.pips_gained, self.params.take_profit_pips)
        
        # Test losing SELL trade
        exit_price = 1.2030  # +30 pips for price, -30 pips for SELL
        result = self.mc_engine._evaluate_trade_outcome(entry_price, exit_price, 'SELL')
        
        self.assertFalse(result.is_win)
        self.assertEqual(result.pips_gained, -self.params.stop_loss_pips)

class TestDataManager(unittest.TestCase):
    """Test data management functionality"""
    
    def setUp(self):
        self.data_manager = DataManager()
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        df = self.data_manager._generate_synthetic_data("EURUSD")
        
        # Should have 20 years of hourly data
        expected_length = 20 * 365 * 24
        self.assertGreater(len(df), expected_length * 0.95)  # Allow for some variation
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check OHLC consistency
        for idx in df.index[:100]:  # Check first 100 rows
            row = df.loc[idx]
            self.assertGreaterEqual(row['high'], max(row['open'], row['close']))
            self.assertLessEqual(row['low'], min(row['open'], row['close']))
            self.assertGreater(row['volume'], 0)
    
    def test_return_calculation(self):
        """Test return calculation and preprocessing"""
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='h')
        prices = np.random.normal(1.2, 0.01, 100)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 0.001,
            'low': prices - 0.001,
            'close': prices,
            'volume': 1000
        }, index=dates)
        
        self.data_manager.price_data = df
        self.data_manager._preprocess_data()
        
        # Check returns calculation
        self.assertEqual(len(self.data_manager.returns), len(df) - 1)
        self.assertAlmostEqual(
            self.data_manager.returns.iloc[0],
            (df['close'].iloc[1] - df['close'].iloc[0]) / df['close'].iloc[0],
            places=6
        )
    
    def test_return_distribution_params(self):
        """Test return distribution parameter calculation"""
        # Create sample returns
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, 1000)
        
        self.data_manager.returns = pd.Series(returns)
        params = self.data_manager.get_return_distribution_params()
        
        required_params = ['mean', 'std', 'skew', 'kurt', 'min', 'max']
        for param in required_params:
            self.assertIn(param, params)
            self.assertIsInstance(params[param], float)

class TestKellyMonteBot(unittest.TestCase):
    """Test the main KellyMonteBot class"""
    
    def setUp(self):
        self.bot = KellyMonteBot(bot_id=0, initial_equity=100000.0)
        self.bot.initialize("EURUSD")
    
    def test_bot_initialization(self):
        """Test bot initialization"""
        self.assertEqual(self.bot.bot_id, 0)
        self.assertEqual(self.bot.initial_equity, 100000.0)
        self.assertEqual(self.bot.current_equity, 100000.0)
        self.assertIsNotNone(self.bot.returns_params)
        self.assertEqual(len(self.bot.trade_history), 0)
    
    def test_signal_generation(self):
        """Test trading signal generation"""
        # Test BUY signal
        market_data = pd.Series({
            'close': 1.2000,
            'sma_20': 1.1980,
            'sma_50': 1.1950
        })
        
        signal = self.bot.generate_trading_signal(market_data)
        self.assertEqual(signal, 'BUY')
        
        # Test SELL signal
        market_data = pd.Series({
            'close': 1.1900,
            'sma_20': 1.1920,
            'sma_50': 1.1950
        })
        
        signal = self.bot.generate_trading_signal(market_data)
        self.assertEqual(signal, 'SELL')
        
        # Test no signal
        market_data = pd.Series({
            'close': 1.1950,
            'sma_20': 1.1950,
            'sma_50': 1.1950
        })
        
        signal = self.bot.generate_trading_signal(market_data)
        self.assertIsNone(signal)
    
    def test_trading_decision_process(self):
        """Test complete trading decision process"""
        current_price = 1.2000
        market_data = pd.Series({
            'close': current_price,
            'sma_20': 1.1980,
            'sma_50': 1.1950,
            'volume': 1000
        })
        timestamp = pd.Timestamp('2024-01-01 12:00:00')
        
        decision = self.bot.make_trading_decision(current_price, market_data, timestamp)
        
        if decision:  # If a decision was made
            self.assertIn('signal', decision)
            self.assertIn('kelly_estimates', decision)
            self.assertIn('position_size', decision)
            self.assertGreater(decision['position_size'], 0)
            self.assertIn(decision['signal'], ['BUY', 'SELL'])
    
    def test_trade_execution_and_closing(self):
        """Test trade execution and closing"""
        # Create a decision
        decision = {
            'timestamp': pd.Timestamp.now(),
            'signal': 'BUY',
            'entry_price': 1.2000,
            'position_size': 1000.0,
            'kelly_estimates': KellyEstimates(
                win_probability=0.6,
                average_win_payoff=2.0,
                average_loss_payoff=1.0,
                payoff_ratio=2.0,
                kelly_fraction=0.4,
                constrained_fraction=0.01
            )
        }
        
        # Execute trade
        trade_result = self.bot.execute_trade(decision)
        
        self.assertEqual(trade_result['signal'], 'BUY')
        self.assertEqual(trade_result['entry_price'], 1.2000)
        self.assertEqual(trade_result['position_size'], 1000.0)
        self.assertEqual(trade_result['status'], 'OPEN')
        self.assertIsNotNone(self.bot.current_position)
        
        # Close trade
        exit_price = 1.2060  # 60 pips profit
        closed_trade = self.bot.close_trade(exit_price, "TAKE_PROFIT")
        
        self.assertEqual(closed_trade['exit_price'], exit_price)
        self.assertEqual(closed_trade['status'], 'CLOSED')
        self.assertGreater(closed_trade['pips_gained'], 0)
        self.assertGreater(closed_trade['pnl'], 0)
        self.assertIsNone(self.bot.current_position)
        self.assertEqual(len(self.bot.trade_history), 1)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Execute and close a few trades
        for i in range(3):
            decision = {
                'timestamp': pd.Timestamp.now(),
                'signal': 'BUY',
                'entry_price': 1.2000,
                'position_size': 1000.0,
                'kelly_estimates': KellyEstimates(
                    win_probability=0.6,
                    average_win_payoff=2.0,
                    average_loss_payoff=1.0,
                    payoff_ratio=2.0,
                    kelly_fraction=0.4,
                    constrained_fraction=0.01
                )
            }
            
            self.bot.execute_trade(decision)
            
            # Alternate between profit and loss
            if i % 2 == 0:
                exit_price = 1.2060  # Profit
            else:
                exit_price = 1.1970  # Loss
            
            self.bot.close_trade(exit_price, "NATURAL")
        
        metrics = self.bot.get_performance_metrics()
        
        self.assertEqual(metrics['bot_id'], 0)
        self.assertEqual(metrics['total_trades'], 3)
        self.assertIn('win_rate', metrics)
        self.assertIn('total_pnl', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_kelly_monte_carlo_integration(self):
        """Test Kelly Criterion with Monte Carlo integration"""
        params = TradingParameters(monte_carlo_scenarios=100)
        bot = KellyMonteBot(bot_id=0, initial_equity=10000.0, params=params)
        bot.initialize("EURUSD")
        
        # Simulate market conditions favoring BUY
        current_price = 1.2000
        market_data = pd.Series({
            'close': current_price,
            'sma_20': 1.1980,
            'sma_50': 1.1950,
            'volume': 1000
        })
        
        decision = bot.make_trading_decision(
            current_price, market_data, pd.Timestamp.now()
        )
        
        if decision:
            # Verify Monte Carlo scenarios were generated
            self.assertIn('mc_scenarios', decision)
            self.assertEqual(decision['mc_scenarios'], 100)
            
            # Verify Kelly estimates are reasonable
            kelly_est = decision['kelly_estimates']
            self.assertGreaterEqual(kelly_est.win_probability, 0.0)
            self.assertLessEqual(kelly_est.win_probability, 1.0)
            self.assertGreaterEqual(kelly_est.constrained_fraction, 0.0)
            self.assertLessEqual(kelly_est.constrained_fraction, params.max_risk_per_trade)
            
            # Verify position sizing
            self.assertGreater(decision['position_size'], 0)
            self.assertLessEqual(
                decision['position_size'], 
                bot.current_equity * params.max_risk_per_trade
            )

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestKellyCalculator,
        TestMonteCarloEngine,
        TestDataManager,
        TestKellyMonteBot,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TESTS COMPLETED")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print(f"\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
