"""
Ray分布式推理实现
"""

import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List
import argparse

import ray
from ray.util.queue import Queue as RayQueue

from config import cfg, parse_cfg_overrides
from dataloader import create_dataloader
from task_tracker import TaskTracker
from workers import DataLoaderWorker, create_worker, QueueMonitor, SaveWorker

# 设置日志
def setup_logging(log_file=None):
    """设置日志配置"""
    if log_file:
        # 重置已有处理器，避免重复日志
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 配置根日志器
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    else:
        logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)


def prepare_data_paths(data_path, output_path, group_by_segment=False):
    """
    准备数据路径列表
    如果输入是目录，返回每个segment文件的路径和对应的输出路径
    如果输入是文件，返回单个文件的路径和原始输出路径
    
    Returns:
        List[Dict]: 每个元素包含 'data_path' 和 'output_path'
    """
    if os.path.isdir(data_path):
        # 目录模式：查找所有 segment*.jsonl 文件
        segment_pattern = os.path.join(data_path, "segment*.jsonl")
        segment_files = glob.glob(segment_pattern)
        
        if not segment_files:
            logger.error(f"No segment*.jsonl files found in {data_path}")
            return []
        
        # 排序确保顺序一致
        segment_files.sort()
        
        logger.info(f"Found {len(segment_files)} segment files:")
        for i, seg_file in enumerate(segment_files):
            logger.info(f"  {i+1}. {os.path.basename(seg_file)}")
        
        # 为每个segment文件准备输出路径
        path_list = []
        for seg_file in segment_files:
            seg_name = os.path.splitext(os.path.basename(seg_file))[0]
            
            if group_by_segment:
                # 分组模式：每个segment有独立的输出目录
                seg_output_path = os.path.join(output_path, seg_name)
            else:
                # 非分组模式：使用原始输出路径
                seg_output_path = output_path
            
            path_list.append({
                'data_path': seg_file,
                'output_path': seg_output_path,
                'segment_name': seg_name
            })
        
        return path_list
    else:
        # 文件模式：直接返回原始路径
        logger.info(f"Input is a single file: {data_path}")
        return [{
            'data_path': data_path,
            'output_path': output_path,
            'segment_name': os.path.splitext(os.path.basename(data_path))[0]
        }]


def generate_all_lance_paths(data_path, output_path, group_by_segment=False):
    """
    生成所有 Lance 数据集的路径列表（不进行分片）
    
    命名规则：path + segment_xx.lance (xx: 00-ff)
    
    Returns:
        List[Dict]: 包含所有 segment 路径的列表
    """
    # 如果传入的是单个 .lance 文件，直接返回
    if data_path.endswith('.lance'):
        segment_name = os.path.splitext(os.path.basename(data_path))[0]
        segment_output = os.path.join(output_path, segment_name) if group_by_segment else output_path
        return [{
            'data_path': data_path,
            'output_path': segment_output,
            'segment_name': segment_name
        }]
    
    # 目录或前缀模式：根据命名规则生成 segment_00.lance ~ segment_ff.lance
    base_path = data_path.rstrip('/') + '/'
    path_list = []
    for i in range(256):
        segment_name = f"segment_{i:02x}"
        segment_file = f"{base_path}{segment_name}.lance"
        segment_output = os.path.join(output_path, segment_name) if group_by_segment else output_path
        path_list.append({
            'data_path': segment_file,
            'output_path': segment_output,
            'segment_name': segment_name
        })
    
    return path_list


def filter_completed_paths(path_list):
    """
    过滤掉已完成的路径（检查 success.jsonl 文件）
    
    Args:
        path_list: 路径列表
        
    Returns:
        List[Dict]: 未完成的路径列表
    """
    remaining_paths = []
    for path_info in path_list:
        output_dir = path_info['output_path']
        success_file = os.path.join(output_dir, 'success.jsonl')
        if os.path.exists(success_file) and os.path.getsize(success_file) > 0:
            logger.info(f"Skipping completed segment: {path_info['segment_name']}")
            continue
        remaining_paths.append(path_info)
    return remaining_paths


def prepare_lance_paths(data_path, output_path, group_by_segment=False, worker_rank: int = 0, world_size: int = 1):
    """
    根据命名规则生成 Lance 数据集的分片路径列表
    
    命名规则：path + segment_xx.lance (xx: 00-ff)
    
    注意：此函数会先过滤掉已完成的路径，然后再进行分片
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got: {world_size}")
    if worker_rank < 0 or worker_rank >= world_size:
        raise ValueError(f"worker_rank must be in [0, {world_size - 1}], got: {worker_rank}")
    
    # 先生成所有路径
    all_paths = generate_all_lance_paths(data_path, output_path, group_by_segment)
    
    # 过滤掉已完成的路径
    remaining_paths = filter_completed_paths(all_paths)
    
    # 对剩余路径进行分片
    total_segments = len(remaining_paths)
    if total_segments == 0:
        return []
    
    chunk_size = total_segments // world_size
    remainder = total_segments % world_size
    
    start = worker_rank * chunk_size + min(worker_rank, remainder)
    end = start + chunk_size + (1 if worker_rank < remainder else 0)
    
    return remaining_paths[start:end]


def run_inference(data_path, output_path, workers, model_path=None, db_path=None, group_id=None, config=None):
    """
    运行分布式推理
    
    Args:
        data_path: 数据路径
        output_path: 输出路径
        workers: Ray workers 列表（必需）
        model_path: 模型路径
        db_path: 任务跟踪文件路径（可选，默认使用 output_path/progress.jsonl）
        group_id: 组ID（用于日志标识）
        config: 配置对象（可选，默认使用全局 cfg）
    """
    # 使用传入的配置或全局配置
    if config is None:
        config = cfg
    
    num_workers = len(workers)

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 设置输出文件路径（output_path 是目录）
    if db_path is None:
        db_path = os.path.join(output_path, 'progress.jsonl')
    log_path = os.path.join(output_path, 'inference.log')

    # 设置日志文件
    setup_logging(log_path)
    
    # 创建数据加载器（根据类型自动选择）
    try:
        dataloader_kwargs = config.get_dataloader_kwargs()
        data_loader = create_dataloader(
            dataloader_type=config.dataloader_type,
            data_path=data_path,
            batch_size=cfg.batch_size,
            **dataloader_kwargs
        )
        total_samples = len(data_loader)
        logger.info(f"Loaded {total_samples} samples")
    except (ValueError, FileNotFoundError, Exception) as e:
        # 检查是否是文件不存在的错误
        error_msg = str(e).lower()
        error_type = type(e).__name__
        # 检查多种文件不存在的错误模式
        is_not_found = (
            'not found' in error_msg or 
            'was not found' in error_msg or 
            'does not exist' in error_msg or
            'no such file' in error_msg or
            (error_type == 'ValueError' and 'not found' in error_msg)
        )
        if is_not_found:
            logger.warning(f"Dataset file not found: {data_path}, skipping...")
            logger.warning(f"Error details: {e}")
            return
        else:
            # 其他错误继续抛出
            logger.error(f"Failed to create dataloader for {data_path}: {e}")
            raise

    # 初始化任务跟踪器（JSONL 版）
    task_tracker = TaskTracker(db_path)
    task_tracker.init_tasks(total_samples)
    # 回收上次中断留下的 allocated 任务，防止"无可分配任务"
    try:
        task_tracker.reset_incomplete_allocations()
    except Exception:
        logger.warning("Failed to reset incomplete allocations; continuing")
    # 获取已完成的任务
    completed_tasks = task_tracker.get_completed_tasks()
    completed_count = len(completed_tasks)
    remaining_tasks = total_samples - completed_count
    logger.info(f"Found {completed_count} completed tasks, resuming from task {completed_count}, remaining: {remaining_tasks}")

    # 创建队列
    input_queue = RayQueue(maxsize=100)  # DataLoader -> ModelWorker
    result_queue = RayQueue(maxsize=1000)  # ModelWorker -> SaveWorker
    
    # 创建多个 DataLoaderWorker（支持并行加载）
    # 使用配置中的 num_dataloader_workers，如果没有设置则默认为 1
    num_loader_workers = getattr(config, 'num_dataloader_workers', 1)
    loader_workers = []
    loader_refs = []
    
    if num_loader_workers > 1 and config.dataloader_type == 'lance':
        # 对于 Lance 数据集，支持数据分片
        samples_per_worker = total_samples // num_loader_workers
        remainder = total_samples % num_loader_workers
        
        logger.info(f"Creating {num_loader_workers} DataLoaderWorkers with data sharding")
        logger.info(f"Total samples: {total_samples}, samples per worker: {samples_per_worker}, remainder: {remainder}")
        
        for i in range(num_loader_workers):
            offset = i * samples_per_worker
            # 最后一个 worker 处理剩余的数据
            limit = samples_per_worker + (remainder if i == num_loader_workers - 1 else 0)
            
            # 创建独立的 dataloader_kwargs（不包含 offset/limit，因为它们作为直接参数传递）
            worker_kwargs = dataloader_kwargs.copy()
            # 移除 offset 和 limit，避免与直接参数冲突
            worker_kwargs.pop('offset', None)
            worker_kwargs.pop('limit', None)
            
            loader_worker = DataLoaderWorker.remote(
                dataloader_type=config.dataloader_type,
                data_path=data_path,
                db_path=db_path,
                batch_size=config.batch_size,
                worker_id=i,
                offset=offset,  # 传递给 DataLoaderWorker.__init__
                limit=limit,    # 传递给 DataLoaderWorker.__init__
                **worker_kwargs,
            )
            loader_workers.append(loader_worker)
            logger.info(f"DataLoaderWorker {i}: offset={offset}, limit={limit} (will process {limit} samples)")
    else:
        # 单个 worker 或非 Lance 数据集
        if num_loader_workers > 1:
            logger.warning(f"Multiple DataLoaderWorkers not supported for {config.dataloader_type}, using 1 worker")
        loader_worker = DataLoaderWorker.remote(
            dataloader_type=config.dataloader_type,
            data_path=data_path,
            db_path=db_path,
            batch_size=config.batch_size,
            worker_id=0,
            **dataloader_kwargs,
        )
        loader_workers.append(loader_worker)
    
    # 创建 SaveWorker
    save_worker = SaveWorker.remote(
        output_path,
        db_path,
        worker_id=0,
        buffer_size=500000,
        log_path=log_path,
    )
    
    # 创建队列监控器
    queue_monitor = QueueMonitor.remote(
        queues={
            'input_queue': input_queue,
            'result_queue': result_queue
        },
        interval=10.0,
        log_path=log_path,
    )
    
    # 启动所有 workers
    # DataLoaderWorker 顺序迭代，不查数据库；SaveWorker 直接调用 mark_tasks_completed，不需要 db_queue
    # 启动所有 DataLoaderWorkers
    for loader_worker in loader_workers:
        loader_refs.append(loader_worker.run.remote(
            input_queue, 
            db_queue=None, 
            num_model_workers=num_workers,
            num_loader_workers=len(loader_workers)
        ))
    model_refs = [w.run.remote(input_queue, result_queue) for w in workers]
    save_ref = save_worker.run.remote(
        result_queue,
        db_queue=None,
        total_tasks=total_samples,
        num_model_workers=num_workers,
    )
    monitor_ref = queue_monitor.run.remote()
    
    # 等待所有 workers 完成
    logger.info("Starting queue pipeline...")
    loader_results = []
    model_results = []
    save_result = 0
    
    print(f"  DataLoaderWorkers: {len(loader_workers)}")
    print(f"  ModelWorkers: {num_workers}")
    print(f"  SaveWorkers: 1")
    print(f"  QueueMonitors: 1")
    print(f"  Total Actors: {len(loader_workers) + num_workers + 2}")
    print(f"  Total samples: {total_samples}")
    print(f"  Batch size: {config.batch_size}")
    print(f"{'='*60}\n")

    try:
        # 一次性等待所有 workers 完成
        save_result = ray.get(save_ref)
        # ray.wait(loader_refs + model_refs, num_returns=1, timeout=4000)
        # all_results = ray.get(loader_refs + model_refs + [save_ref],timeout=1500)
        # 停止队列监控器
        ray.cancel(monitor_ref)
        # 分离结果：loader_results, model_results, save_result
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        ray.cancel(monitor_ref)
        raise

    # 显示最终统计信息
    stats = task_tracker.get_progress_stats()
    logger.info(f"Final stats: {stats}")
    
    total_model_batches = sum(model_results) if model_results else 0
    total_saved = save_result if isinstance(save_result, int) else 0
    logger.info(f"Completed: saved={total_saved} items, model_batches={total_model_batches}, num_model_workers={len(model_refs)}")
    logger.info(f"Results saved to: {os.path.join(output_path, 'results.jsonl')}")
    # 写入成功标记文件，表示该输出目录内的数据已全部处理完成
    success_path = os.path.join(output_path, 'success.jsonl')
    try:
        with open(success_path, 'w', encoding='utf-8') as sf:
            sf.write(json.dumps({
                'status': 'success',
                'total_saved': total_saved,
                'loader_batches': sum(loader_results) if loader_results else 0,
                'model_batches': total_model_batches,
                'num_workers': len(model_refs)
            }, ensure_ascii=False) + '\n')
        logger.info(f"Success marker saved to: {success_path}")
    except Exception as e:
        logger.error(f"Failed to write success marker: {e}")
    logger.info(f"Progress saved to: {db_path}")
    logger.info(f"Logs saved to: {log_path}")
    
    
if __name__ == '__main__':
    # --- 常用配置默认值（可以被命令行覆盖） ---
    cfg.data_path = 's3://embodied-multimodality/speech/processed_datasets/stage1/audio_b1_4_0/'  # 数据或目录
    cfg.output_path = './outputs'  # 推理结果目录
    cfg.model_type = 'beats'  # 可选: 'beats', 'qwen3omni', 'chordino', 'essentia', 'beatnet', 'music_cpu_pipeline'
    cfg.dataloader_type = 'lance'  # 可选: 'oss', 'audio_jsonl', 'jsonl', 'lance'
    cfg.lance_prompt_key = "audio_flac"
    cfg.batch_size = 128
    cfg.num_dataloader_workers = 10
    cfg.gpu_per_worker = 0.05  # 自动计算 num_workers 时使用
    cfg.num_workers = 0  # 设为正整数即可固定 worker 数量
    cfg.group_by_segment = True
    
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser()
    # rank / world size 属于执行环境，而不是 cfg 本身
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    # 通用 cfg 覆盖：可以多次传入
    parser.add_argument(
        "--cfg",
        action="append",
        default=[],
        help="覆盖 cfg 中的任意字段，如: --cfg model_type=qwen3omni --cfg batch_size=64",
    )

    args = parser.parse_args()

    # 通用 --cfg 覆盖（自动按类型转换）
    if args.cfg:
        overrides = parse_cfg_overrides(args.cfg)
        cfg.update(**overrides)

    worker_rank = args.worker_rank
    world_size = args.world_size

    # 使用（可能被覆盖后的）配置中的数据路径
    data_path = cfg.data_path
    output_path = cfg.output_path
    group_by_segment = cfg.group_by_segment

    if cfg.dataloader_type != 'lance' and not os.path.exists(data_path):
        print(f"Error: Data path not found: {data_path}")
        sys.exit(1)

    # 准备数据路径列表
    if cfg.dataloader_type == 'lance':
        # 先生成所有路径并过滤已完成的，然后按 worker_rank 和 world_size 分片
        all_paths = generate_all_lance_paths(data_path, output_path, group_by_segment)
        remaining_paths = filter_completed_paths(all_paths)
        logger.info(
            "Lance paths: total=%s, completed=%s, remaining=%s",
            len(all_paths),
            len(all_paths) - len(remaining_paths),
            len(remaining_paths),
        )
        
        # 对剩余路径进行分片
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got: {world_size}")
        if worker_rank < 0 or worker_rank >= world_size:
            raise ValueError(f"worker_rank must be in [0, {world_size - 1}], got: {worker_rank}")
        
        total_segments = len(remaining_paths)
        if total_segments == 0:
            path_list = []
        else:
            chunk_size = total_segments // world_size
            remainder = total_segments % world_size
            start = worker_rank * chunk_size + min(worker_rank, remainder)
            end = start + chunk_size + (1 if worker_rank < remainder else 0)
            path_list = remaining_paths[start:end]
        
        logger.info(
            "Lance sharding: worker_rank=%s, world_size=%s, assigned_segments=%s",
            worker_rank,
            world_size,
            len(path_list),
        )
    else:
        path_list = prepare_data_paths(data_path, output_path, group_by_segment)
    print(path_list)
    
    if not path_list:
        print("No valid data paths found")
        sys.exit(1)
    
    # 初始化 Ray 与预创建 workers（仅一次）
    ray.init()
    available_resources = ray.available_resources()
    available_gpus = available_resources.get('GPU', 0)
    
    # 根据配置确定 worker 数量（优先使用显式 num_workers）
    gpu_per_worker = getattr(cfg, 'gpu_per_worker', 0.05) or 0.05
    num_workers = cfg.num_workers or (max(1, int(available_gpus / gpu_per_worker)) if gpu_per_worker > 0 else 1)
    logger.info(f"num_workers={num_workers}, gpus={available_gpus}, gpu_per_worker={gpu_per_worker}, configured={cfg.num_workers}")
    
    workers = []
    print("num_workers:", num_workers)
    model_path = getattr(cfg, "model_path", None) or None  # 允许某些模型不需要 model_path
    for i in range(num_workers):
        workers.append(create_worker(model_type=cfg.model_type, model_path=model_path))
    # 等待所有 worker 完成初始化
    try:
        ray.get([w.get_model_info.remote() for w in workers])
        logger.info("All workers are ready")
    except Exception as e:
        logger.error(f"Waiting workers ready failed: {e}")
        raise

    # 直接使用所有 workers 处理每个 segment
    for path_info in path_list:
        data_file = path_info['data_path']
        output_dir = path_info['output_path']
        segment_name = path_info['segment_name']

        print(f"\n{'='*60}")
        print(f"Processing: {segment_name}")
        print(f"Data file: {data_file}")
        print(f"Output dir: {output_dir}")
        print(f"{'='*60}")

        # 双重保险：再次检查 success 文件（虽然已经在分片前过滤过）
        success_file = os.path.join(output_dir, 'success.jsonl')
        if os.path.exists(success_file) and os.path.getsize(success_file) > 0:
            print(f"{segment_name} already completed, skipping")
            continue

        run_inference(
            data_path=data_file,
            output_path=output_dir,
            workers=workers,
            model_path=model_path,
            group_id=None,
            config=cfg,
        )
        print(f"{segment_name} completed successfully")
    
    print(f"\n{'='*60}")
    print("All segments processed!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    ray.shutdown()
