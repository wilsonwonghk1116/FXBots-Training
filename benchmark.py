"""
Performance benchmark for technical indicators
"""
from src.env import SmartForexEnvironment
from src.utils import timeit
import numpy as np

# 生成测试数据
prices = np.cumprod(1 + np.random.normal(0, 0.001, 10000))
highs = prices * 1.001
lows = prices * 0.999
volumes = np.random.randint(100, 10000, len(prices))

env = SmartForexEnvironment()

@timeit
def benchmark_indicators():
    for i in range(100, len(prices)):
        env._intensive_market_analysis(prices[i])

if __name__ == "__main__":
    print("Running benchmark...")
    benchmark_indicators()
    print("Benchmark completed")