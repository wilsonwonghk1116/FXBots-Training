import pytest
from src.env import SmartForexEnvironment

def test_env_reset():
    env = SmartForexEnvironment()
    obs, info = env.reset()
    assert obs.shape == (20,)
    assert isinstance(info, dict)
