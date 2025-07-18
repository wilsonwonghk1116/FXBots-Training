import os
import psutil
import time
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventType

# Add CPU temperature threshold and helper
CPU_TEMP_THRESHOLD = float(os.environ.get("CPU_TEMP_THRESHOLD", "80.0"))

def get_max_cpu_temp() -> float:
    """
    Returns the maximum current CPU core temperature in Celsius.
    """
    temps = psutil.sensors_temperatures()
    max_temp = 0.0
    for entries in temps.values():
        for entry in entries:
            if getattr(entry, "current", None) is not None and entry.current > max_temp:
                max_temp = entry.current
    return max_temp

def cpu_worker_process(shutdown_event: EventType, target_usage: float, core_id: int):
    """
    A function designed to be run in a separate process to consume CPU.
    It runs in a busy-wait loop to achieve the target usage percentage.
    This method effectively targets a percentage of CPU time, which translates
    to the desired utilization regardless of the CPU's clock speed (e.g., 3.6GHz or 4.9GHz).
    """
    # This print statement is a diagnostic tool to confirm the process has spawned.
    print(f"DIAGNOSTIC (PID: {os.getpid()}): CPU worker process successfully spawned for core {core_id}.", flush=True)

    p = psutil.Process(os.getpid())
    try:
        # Pin process to a specific CPU core for better control
        if hasattr(p, 'cpu_affinity'):
            p.cpu_affinity([core_id])
    except psutil.NoSuchProcess:
        return # Process may have been terminated

    # Begin saturation loop with thermal safeguard
    while not shutdown_event.is_set():
        # Thermal check: stop if CPU temp exceeds threshold
        current_temp = get_max_cpu_temp()
        if current_temp > CPU_TEMP_THRESHOLD:
            print(f"DIAGNOSTIC (PID: {os.getpid()}): CPU temp {current_temp:.1f}°C exceeds threshold {CPU_TEMP_THRESHOLD}°C. Exiting.", flush=True)
            break
        start_time = time.time()
        # Busy-wait loop to consume CPU
        while (time.time() - start_time) < target_usage:
            pass # Actively consume CPU time
        # Sleep to yield the CPU, completing the cycle
        time.sleep(1 - target_usage) 