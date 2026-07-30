import torch

def detached_integral(u, detach_window_size):
    # u: (B, T=80, D)
    cum_detach = torch.cumsum(u.detach(), dim=-2)
    cum_normal = torch.cumsum(u, dim=-2)

    # number of gradient from previous timesteps contained in: 
    # shifted: [0, 1, 2, ..., window_size-1, window_size, ...., T] ->
    # shifted: [T-window_size+1, T-window_size+2, ...,T, 0, 1, 2, ...., T - window_size] ->
    # sum_recent: [0, 1, 2, ..., window_size-1, window_size, ...., window_size]
    shifted = torch.roll(cum_normal, shifts=detach_window_size, dims=-2)
    shifted[:, :, :detach_window_size] = 0
    sum_recent = cum_normal - shifted
        
    cum_detach_shifted = torch.roll(cum_detach, shifts=detach_window_size, dims=-2)
    cum_detach_shifted[:, :, :detach_window_size] = 0
        
    cumulative_sum = cum_detach_shifted + sum_recent
    return cumulative_sum