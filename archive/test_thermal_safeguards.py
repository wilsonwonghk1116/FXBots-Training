import unittest
import os
from saturator_utils import get_max_cpu_temp, CPU_TEMP_THRESHOLD, cpu_worker_process
import psutil
from unittest import mock

class TestCpuThermalSafeguard(unittest.TestCase):
    @mock.patch('psutil.sensors_temperatures')
    def test_get_max_cpu_temp(self, mock_sensors):
        mock_entry = mock.Mock(current=90.5)
        mock_sensors.return_value = {'coretemp': [mock_entry]}
        temp = get_max_cpu_temp()
        self.assertEqual(temp, 90.5)
        self.assertTrue(temp > CPU_TEMP_THRESHOLD)

    @mock.patch('saturator_utils.get_max_cpu_temp', return_value=85.0)
    def test_cpu_worker_process_exits(self, mock_temp):
        event = mock.Mock()
        event.is_set.side_effect = [False]
        # The process should exit immediately due to high temperature
        cpu_worker_process(event, target_usage=0.1, core_id=0)
        self.assertTrue(mock_temp.called)

class TestEnvironmentVariables(unittest.TestCase):
    def test_gpu_threshold_env(self):
        os.environ['GPU_TEMP_THRESHOLD'] = '75.0'
        import importlib
        module = importlib.reload(__import__('run_dual_pc_training_standalone'))
        self.assertEqual(module.GPU_TEMP_THRESHOLD, 75.0)

if __name__ == '__main__':
    unittest.main() 