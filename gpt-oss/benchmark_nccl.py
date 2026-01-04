import os
import socket
import torch
import torch.distributed as dist

# 转换字节数为人类可读格式
def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"

def run_benchmark():
    # 1. 获取环境信息
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # 2. 设置设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 3. 初始化分布式环境 (NCCL)
    # 注意：多机环境下，这一步会阻塞，直到所有机器都连上 Master 端口
    dist.init_process_group(backend="nccl")

    # 4. 验证多机连接：打印每台机器的名字
    hostname = socket.gethostname()
    # 创建一个张量来收集所有 rank 的主机名（稍微繁琐点，为了可视化）
    # 这里简单点，直接让每个节点打印一下
    if local_rank == 0:
        print(f"✅ 节点连接成功: [Rank {rank}] 位于主机 {hostname}")
    
    # 确保所有人都打印完
    dist.barrier()

    if rank == 0:
        print(f"\n================================================================")
        print(f"多机 NCCL 测速 | 总卡数 (World Size): {world_size}")
        print(f"================================================================")
        print(f"{'Size':<15} | {'Avg Time (ms)':<15} | {'Bus Bandwidth (GB/s)':<25}")
        print(f"----------------------------------------------------------------")

    # 5. 定义测试参数 (测试包从 128MB 到 1GB)
    sizes_mb = [128, 256, 512, 1024] 
    num_iters = 20
    warmup = 5

    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        num_elements = int(size_bytes / 4)
        tensor = torch.randn(num_elements, device=device, dtype=torch.float32)
        
        # 热身
        for _ in range(warmup):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event) / num_iters

        # 计算带宽
        # 环算法带宽公式: Size * 2 * (n-1)/n / Time
        correction_factor = 2 * (world_size - 1) / world_size
        alg_bandwidth_gbs = (size_bytes * correction_factor) / (elapsed_time_ms / 1000.0) / (1024**3)

        if rank == 0:
            print(f"{format_size(size_bytes):<15} | {elapsed_time_ms:<15.4f} | {alg_bandwidth_gbs:<25.2f}")

    dist.destroy_process_group()


"""
torchrun --nproc_per_node=8 --node_rank=0 --master_port=6000 --master_addr='10.45.175.69' --nnodes=2 benchmark_nccl.py
torchrun --nproc_per_node=8 --node_rank=1 --master_port=6000 --master_addr='10.45.175.69' --nnodes=2 benchmark_nccl.py
"""

if __name__ == "__main__":
    run_benchmark()
