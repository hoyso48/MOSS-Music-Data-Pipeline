# -*- coding: utf-8 -*-
"""
SaveWorker: 独立的保存 Worker，异步处理文件写入和数据库更新
"""
import json
import logging
import os
import time
from typing import Dict, List, Any

import ray
from ray.util.queue import Queue as RayQueue

from task_tracker import TaskTracker

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class SaveWorker:
    """独立的保存 Worker，从队列读取结果并保存到文件和数据库"""
    
    def __init__(
        self,
        output_path: str,
        db_path: str,
        worker_id: int = 0,
        buffer_size: int = 64,
        progress_interval: float = 2.0,
        log_path: str = None,
    ):
        """
        初始化 SaveWorker
        
        Args:
            output_path: 输出目录路径
            db_path: 数据库路径
            worker_id: Worker ID（用于多 worker 场景）
            buffer_size: 缓冲区大小，达到此数量时批量写入
            progress_interval: 进度报告间隔（秒），默认 10.0
            log_path: 推理日志文件路径，用于写入进度信息
        """
        if log_path:
            self._configure_logger(log_path)
        
        self.output_path = output_path
        self.db_path = db_path
        self.worker_id = worker_id
        self.buffer_size = buffer_size
        self.progress_interval = progress_interval
        
        # 文件路径
        self.jsonl_path = os.path.join(output_path, 'results.jsonl')
        os.makedirs(os.path.dirname(self.jsonl_path) or '.', exist_ok=True)
        
        # 数据库跟踪器（SaveWorker 独占，负责所有数据库写入）
        self.task_tracker = TaskTracker(db_path)
        
        # 写入缓冲区
        self.write_buffer: List[Dict[str, Any]] = []
        self.task_ids_buffer: List[int] = []
        
        # 进度统计
        self.total_saved = 0
        self.initial_completed = 0  # resume时已完成的任务数
        self.start_time = None
        self.last_progress_time = None
        self.time_stats = {
            'serialize': 0.0,
            'write': 0.0,
            'tracker': 0.0,
            'queue_wait': 0.0,
        }
        
        logger.info(f"SaveWorker {worker_id} initialized, output: {self.jsonl_path}")
    
    def _configure_logger(self, log_path: str):
        """为 Ray Actor 配置文件日志（仅配置一次）"""
        if any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_path)
               for h in logger.handlers):
            return
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    def _write_to_file(self, batch_results: List[Dict[str, Any]]):
        """写入文件（追加模式，批量写入以提高效率）"""
        # 批量构建 JSON 字符串，减少 I/O 次数
        lines = []
        for result in batch_results:
            # 移除临时URL字段和audio_bytes字段，避免将presigned URL和大量二进制数据持久化
            safe_result = dict(result) if isinstance(result, dict) else result
            if isinstance(safe_result, dict):
                safe_result.pop('url', None)
                # 确保移除 audio_bytes（可能在 to_dict 中已移除，但这里再次确保）
                safe_result.pop('audio_bytes', None)  # 移除音频 bytes 数据，节省存储空间
                # 递归检查嵌套字典中是否有 bytes 类型
                safe_result = self._remove_bytes_from_dict(safe_result)
            try:
                lines.append(json.dumps(safe_result, ensure_ascii=False) + '\n')
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize result: {e}, keys: {list(safe_result.keys()) if isinstance(safe_result, dict) else 'N/A'}")
                # 尝试再次清理
                safe_result = {k: v for k, v in safe_result.items() if not isinstance(v, bytes)}
                try:
                    lines.append(json.dumps(safe_result, ensure_ascii=False) + '\n')
                except Exception as e2:
                    logger.error(f"Failed to serialize after cleanup: {e2}, skipping this result")
                    continue
        
        # 一次性写入所有行
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.writelines(lines)
    
    def _remove_bytes_from_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """递归移除字典中的 bytes 类型值"""
        result = {}
        for k, v in d.items():
            if isinstance(v, bytes):
                continue  # 跳过 bytes 类型
            elif isinstance(v, dict):
                result[k] = self._remove_bytes_from_dict(v)
            elif isinstance(v, list):
                result[k] = [self._remove_bytes_from_dict(item) if isinstance(item, dict) else item 
                            for item in v if not isinstance(item, bytes)]
            else:
                result[k] = v
        return result
    
    def _flush_buffer(self):
        """刷新缓冲区：写入文件并更新数据库。返回 count"""
        if not self.write_buffer:
            return 0
        
        try:
            # 写入文件
            write_start = time.time()
            self._write_to_file(self.write_buffer)
            self.time_stats['write'] += time.time() - write_start
            
            # 更新数据库
            if self.task_ids_buffer:
                tracker_start = time.time()
                self.task_tracker.mark_tasks_completed(self.task_ids_buffer)
                self.time_stats['tracker'] += time.time() - tracker_start
            
            count = len(self.write_buffer)
            self.write_buffer.clear()
            self.task_ids_buffer.clear()
            return count
        except Exception as e:
            logger.error(f"SaveWorker {self.worker_id} flush buffer failed: {e}")
            raise
    
    def _log_progress(self, total_tasks: int = None):
        """记录进度信息"""
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        if total_tasks is None:
            # 尝试从数据库获取总数
            try:
                stats = self.task_tracker.get_progress_stats()
                total_tasks = stats.get('completed', 0) + stats.get('allocated', 0) + stats.get('unallocated', 0)
            except Exception:
                total_tasks = None
        
        # 计算总进度：包括初始已完成的任务和新保存的任务
        total_completed = self.initial_completed + self.total_saved
        
        if total_tasks and total_tasks > 0:
            progress_pct = (total_completed / total_tasks) * 100 if total_tasks > 0 else 0
            speed = self.total_saved / elapsed if elapsed > 0 else 0
            eta_seconds = (total_tasks - total_completed) / speed if speed > 0 else 0
            eta_minutes = eta_seconds / 60
            timing_msg = ""
            if elapsed > 0:
                write_pct = (self.time_stats['write'] / elapsed) * 100
                tracker_pct = (self.time_stats['tracker'] / elapsed) * 100
                serialize_pct = (self.time_stats['serialize'] / elapsed) * 100
                queue_pct = (self.time_stats['queue_wait'] / elapsed) * 100
                other_pct = max(0.0, 100.0 - (write_pct + tracker_pct + serialize_pct + queue_pct))
                timing_msg = (
                    f" | Time split: serialize={serialize_pct:.1f}%, "
                    f"write={write_pct:.1f}%, tracker={tracker_pct:.1f}%, "
                    f"queue_wait={queue_pct:.1f}%, other={other_pct:.1f}%"
                )
            
            msg = (
                f"[SaveWorker {self.worker_id}] Progress: {total_completed}/{total_tasks} "
                f"({progress_pct:.1f}%) | "
                f"Time: {elapsed:.1f}s | "
                f"Speed: {speed:.1f} samples/s | "
                f"ETA: {eta_minutes:.1f}min"
                f"{timing_msg}"
            )
        else:
            speed = self.total_saved / elapsed if elapsed > 0 else 0
            msg = (
                f"[SaveWorker {self.worker_id}] Progress: {total_completed} completed ({self.total_saved} new) | "
                f"Time: {elapsed:.1f}s | "
                f"Speed: {speed:.1f} samples/s"
            )
        
        logger.info(msg)
    
    def run(
        self,
        result_queue: RayQueue,
        db_queue: RayQueue = None,
        total_tasks: int = None,
        num_model_workers: int = 1,
    ):
        """
        从结果队列读取并保存
        
        Args:
            result_queue: 结果队列，每个元素为 (batch_results: List[AudioInfo], task_ids: List[int]) 或 None
            db_queue: 数据库操作队列（已废弃，保留以兼容接口）
            total_tasks: 总任务数（用于进度计算），如果为 None 则从数据库查询
            num_model_workers: 模型 worker 数量，用于接收全部结束信号
        
        Returns:
            保存的总数量
        """
        # 在开始前获取已完成的任务数（用于resume场景的进度计算）
        try:
            completed_tasks = self.task_tracker.get_completed_tasks()
            self.initial_completed = len(completed_tasks)
        except Exception:
            self.initial_completed = 0
        
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        end_signals_received = 0
        
        # 输出启动信息
        logger.info(
            "[SaveWorker %s] Started, total_tasks=%s, initial_completed=%s",
            self.worker_id,
            total_tasks,
            self.initial_completed,
        )
        
        while True:
            try:
                # 处理结果队列（阻塞等待）
                try:
                    queue_start = time.time()
                    item = result_queue.get(timeout=60.0)  # 设置超时时间
                    self.time_stats['queue_wait'] += time.time() - queue_start
                except TimeoutError:
                    # 超时时返回 None，而不是抛出异常
                    self.time_stats['queue_wait'] += time.time() - queue_start
                    item = None
                except Exception as e:
                    logger.warning(f"[SaveWorker] Queue.get() error: {e}")
                    self.time_stats['queue_wait'] += time.time() - queue_start
                    item = None
                
                # 结束标记
                if item is None :
                    # 刷新剩余缓冲区
                    if self.write_buffer:
                        count = self._flush_buffer()
                        self.total_saved += count
                    end_signals_received += 1
                    stats = self.task_tracker.get_progress_stats()
                    total_tasks = stats.get('completed', 0) + stats.get('allocated', 0) + stats.get('unallocated', 0)
                    gap=abs(self.initial_completed + self.total_saved - total_tasks)
                    print("gap:",gap,"end_signals_received:",end_signals_received)
                    # if end_signals_received >= max(1, num_model_workers) or (gap == 0):
                    if gap == 0:
                        print("saver bye bye")
                        break
                    else:
                        continue
                
                batch_results, task_ids = item
                if not batch_results:
                    continue
                
                # 过滤掉解码失败的样本，并保持 task_ids 和 results 的对应关系
                successful_results = []
                successful_task_ids = []
                serialize_start = time.time()
                for idx, result in enumerate(batch_results):
                    if hasattr(result, 'error') and result.error:
                        # 跳过解码失败的样本，不加入缓冲区，也不标记为 completed
                        continue
                    # 转换为字典以便保存
                    if hasattr(result, 'to_dict'):
                        result_dict = result.to_dict()
                    else:
                        result_dict = dict(result)

                    if isinstance(result_dict, dict):
                        # 移除不需要保存的字段
                        result_dict.pop('audio_type', None)
                        extra = result_dict.get('_extra')
                        if isinstance(extra, dict):
                            extra.pop('audio_type', None)

                    successful_results.append(result_dict)
                    # 保持 task_ids 和 results 的对应关系
                    if task_ids and idx < len(task_ids):
                        successful_task_ids.append(task_ids[idx])
                self.time_stats['serialize'] += time.time() - serialize_start
                
                # 添加到缓冲区（只保存成功的样本和对应的 task_ids）
                if successful_results:
                    self.write_buffer.extend(successful_results)
                    if successful_task_ids:
                        self.task_ids_buffer.extend(successful_task_ids)
                
                # 达到缓冲区大小时刷新（包括数据库更新）
                if len(self.write_buffer) >= self.buffer_size:
                    count = self._flush_buffer()
                    self.total_saved += count
                    # 刷新后，task_ids_buffer 已在 _flush_buffer 中清空并标记为 completed
                
                # 定期输出进度信息（即使 total_saved 为 0 也要输出，确保监控可见）
                current_time = time.time()
                if current_time - self.last_progress_time >= self.progress_interval:
                    count = self._flush_buffer()
                    self.total_saved += count
                    self._log_progress(total_tasks)
                    self.last_progress_time = current_time
                    if abs(self.initial_completed + self.total_saved - total_tasks)==0:
                        print("saver bye bye")
                        break
                    
            except Exception as e:
                logger.error(f"SaveWorker {self.worker_id} run error: {e}")
                continue
        
        # 最终进度报告
        # 结束前做最后一次刷新
        count = self._flush_buffer()
        self.total_saved += count
        self._log_progress(total_tasks)
        logger.info(f"SaveWorker {self.worker_id} completed, total saved: {self.total_saved}")
        return self.total_saved

