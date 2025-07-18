import ray
ray.init(address='auto')
@ray.remote(num_gpus=1)
def gpu_test():
    import torch
    return torch.cuda.is_available(), torch.cuda.get_device_name(torch.cuda.current_device())
futures = [gpu_test.remote() for _ in range(2)]
print(ray.get(futures))
