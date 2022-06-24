import torch
import gc

def get_default_device():
    """ Pick GPU if available, else CPU """
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    else:
        return torch.device('cpu')

def to_device(data, device):
    """ Move tensor(s) to chosen device """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    
    if isinstance(data, dict):
        return {k: to_device(t, device) for k, t in data.items()}
    
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """ Wrap a dataloader to move data to a device """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """ Yield a batch of data after moving it to device """
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """ Number of batches """
        return len(self.dl)

def get_cuda_memory_info(device: torch.device) -> None:
    """ Get the memory info for a cuda device (allocated memory is a subset of reserved) """
    if device == torch.device(type="cuda"):
        total = round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2)
        reserved = round(torch.cuda.memory_reserved(device) / 2**30, 2)
        allocated = round(torch.cuda.memory_allocated(device) / 2**30, 2)
        available = round(total - reserved, 2)
        print(f"CUDA MEMORY STATISTICS")
        print('-' * 30)
        print(f"Total     : {total} GB")
        print(f"Reserved  : {reserved} GB")
        print(f"Allocated : {allocated} GB")
        print(f"Available : {available} GB")
        
    else:
        print("cuda not in use")

def clear_cache(device: torch.device) -> None:
    if device == torch.device(type="cuda"):
        reserved = round(torch.cuda.memory_reserved(device) / 2**30, 2)
        allocated_before = round(torch.cuda.memory_allocated(device) / 2**30, 2)
        gc.collect()
        torch.cuda.empty_cache()
        allocated_after = round(torch.cuda.memory_allocated(device) / 2**30, 2)

        print(f"{round(allocated_before - allocated_after, 2)} GB allocated memory cleared")
              
    else:
        print("cuda not in use")

def clear_cache_and_get_info(device: torch.device) -> None:
    if device == torch.device(type="cuda"):
        clear_cache(device)
        get_cuda_memory_info(device)
              
    else:
        print("cuda not in use")