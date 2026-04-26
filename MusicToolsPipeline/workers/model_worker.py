# -*- coding: utf-8 -*-
"""
Ray Model Worker
简化的模型工作器，只需要继承基础模型类即可
"""
import ray
import logging
from typing import List, Dict, Any, Union
from ray.util.queue import Queue as RayQueue
from audio_info import AudioInfo

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=0.05,num_cpus=0.5)
class ModelWorker:
    """
    Ray Actor，每个GPU一个worker
    简化的实现，只需要指定模型类即可
    """
    
    def __init__(self, model_class, model_name: str, model_path: str = None, **kwargs):
        """
        初始化模型工作器
        
        Args:
            model_class: 模型类（如BEATsModel）
            model_name: 模型名称
            model_path: 模型路径
            **kwargs: 其他参数
        """
        self.model_class = model_class
        self.model_name = model_name
        self.model_path = model_path
        self.kwargs = kwargs
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            # 创建模型实例，让模型类自己处理设备分配
            self.model = self.model_class(
                model_name=self.model_name,
                model_path=self.model_path,
                **self.kwargs
            )
            logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_batch(self, batch_data: List[AudioInfo]) -> List[AudioInfo]:
        """
        处理一批数据
        
        Args:
            batch_data: AudioInfo 对象列表
            
        Returns:
            AudioInfo 对象列表（包含预测结果）
        """
        try:
            # 使用模型的generate_batch方法
            results = self.model.generate_batch(batch_data)
            
            # 添加worker信息到 _extra 字段（用于向后兼容）
            for result in results:
                if not hasattr(result, '_extra'):
                    result._extra = {}
                result._extra["worker_processed"] = True
                # 使用 model_name 作为 worker 标识（如果没有 idx 属性）
                worker_id = getattr(self, 'idx', None) or getattr(self, 'model_name', 'unknown')
                result._extra["worker_id"] = worker_id
            
            return results
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model:
            info = self.model.get_model_info()
            info["worker_name"] = self.model_name
            return info
        return {"worker_name": self.model_name, "status": "unloaded"}
    
    def run(self, input_queue: RayQueue, result_queue: RayQueue):
        """
        从输入队列读取数据，处理，放入结果队列
        
        Args:
            input_queue: 输入队列，每个元素为 (batch_data: List[AudioInfo], task_ids: List[int]) 或 None
            result_queue: 结果队列，放入处理后的 AudioInfo 批次
        
        Returns:
            处理的总批次数量
        """
        batch_count = 0
        
        while True:
            try:
                item = input_queue.get()
                
                # 结束标记
                if item is None:
                    # 持续尝试从队列中获取实际数据，直到遇到非 None 或队列耗尽
                    queue_size = input_queue.qsize()
                    if queue_size > 0:
                        print(
                            f"ModelWorker {self.model_name} received end signal but queue still has "
                            f"{queue_size} items, trying to drain remaining items..."
                        )

                    non_none_item = None
                    pending = queue_size
                    while pending > 0:
                        next_item = input_queue.get()
                        if next_item is None:
                            # 保留结束信号供其他 worker 使用
                            pending -= 1
                            input_queue.put(None)
                            continue
                            
                            
                        else:
                            non_none_item = next_item
                            break

                    if non_none_item:
                        item = non_none_item
                    else:
                        # 队列中没有实际数据了，把结束信号放回尾部并结束
                        # input_queue.put(None)
                        result_queue.put(None)
                        print("woker byebye")
                        break
                
                batch_data, task_ids = item
                
                if not batch_data:
                    continue
                
                # 处理批次
                try:
                    results = self.generate_batch(batch_data)
                    # 放入结果队列
                    result_queue.put((results, task_ids))
                    batch_count += 1
                except Exception as e:
                    logger.error(f"Error processing batch {task_ids}: {e}")
                    # 即使失败也放入结果队列（带错误标记）
                    error_results = []
                    for audio_info in batch_data:
                        audio_info.error = str(e)
                        audio_info.predictions = []
                        error_results.append(audio_info)
                    result_queue.put((error_results, task_ids))
                    continue
                    
            except Exception as e:
                logger.error(f"ModelWorker {self.model_name} run error: {e}")
                continue
        
        logger.info(f"ModelWorker {self.model_name} completed, processed {batch_count} batches")
        return batch_count
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            self.model.cleanup()
            self.model = None


def create_model_worker(model_class, model_name: str, model_path: str = None, **kwargs):
    """创建通用模型工作器"""
    return ModelWorker.remote(model_class, model_name, model_path, **kwargs)



def create_worker(model_type: str = None, model_name: str = None, model_path: str = None, **kwargs):
    """
    工厂函数：根据类型创建对应的模型工作器
    
    Args:
        model_type: 模型类型，可选值：'beats', 'qwen3omni', 'chordino', 'essentia',
                    'essentia_instrument', 'beatnet', 'music_cpu_pipeline',
                    'music_cpu_lite_pipeline'
                   如果为 None，从环境变量 MODEL_TYPE 读取
        model_name: 模型名称
        model_path: 模型路径
        **kwargs: 其他参数（传递给具体的模型）
        
    Returns:
        ModelWorker 实例
    """
    import os
    from sub_models import (
        BEATsModel,
        # Qwen3OmniModel,
        ChordinoModel,
        EssentiaModel,
        EssentiaInstrumentModel,
        BeatNetModel,
        MusicCpuPipelineModel,
        MusicCpuLitePipelineModel,
        DASMModel,
    )

    if model_type is None:
        model_type = os.environ.get('MODEL_TYPE', 'beats').lower()

    model_map = {
        'beats': {
            'cls': BEATsModel,
            'default_name': 'BEATs',
            'options': {'num_gpus': 0.05, 'num_cpus': 0.5},
        },
        # 'qwen3omni': {
        #     'cls': Qwen3OmniModel,
        #     'default_name': 'Qwen3-Omni-30B-A3B-Instruct',
        #     'options': None,
        # },
        'chordino': {
            'cls': ChordinoModel,
            'default_name': 'Chordino',
            'options': {'num_gpus': 0, 'num_cpus': kwargs.pop('num_cpus', 0.3)},
        },
        'essentia': {
            'cls': EssentiaModel,
            'default_name': 'Essentia',
            'options': {'num_gpus': 0, 'num_cpus': kwargs.pop('num_cpus', 0.3)},
        },
        'essentia_instrument': {
            'cls': EssentiaInstrumentModel,
            'default_name': 'EssentiaInstrument',
            'options': {'num_gpus': 0, 'num_cpus': kwargs.pop('num_cpus', 2.0)},
        },
        'instrument': {
            'cls': EssentiaInstrumentModel,
            'default_name': 'EssentiaInstrument',
            'options': {'num_gpus': 0, 'num_cpus': kwargs.pop('num_cpus', 2.0)},
        },
        'beatnet': {
            'cls': BeatNetModel,
            'default_name': 'BeatNet',
            'options': {'num_gpus': 0, 'num_cpus': kwargs.pop('num_cpus', 0.3)},
        },
        'music_cpu_pipeline': {
            'cls': MusicCpuPipelineModel,
            'default_name': 'MusicCpuPipeline',
            'options': {'num_gpus': 0, 'num_cpus': kwargs.pop('num_cpus', 0.1)},
        },
        'music_cpu_lite_pipeline': {
            'cls': MusicCpuLitePipelineModel,
            'default_name': 'MusicCpuLitePipeline',
            'options': {'num_gpus': 0, 'num_cpus': kwargs.pop('num_cpus', 0.1)},
        },
        'dasm': {
            'cls': DASMModel,
            'default_name': 'DASM',
            'options': {
                'num_gpus': kwargs.pop('num_gpus', 0.5),
                'num_cpus': kwargs.pop('num_cpus', 1.0),
            },
        },
    }

    if model_type not in model_map:
        raise ValueError(
            f"Unknown model_type: {model_type}. Supported: "
            f"'beats', 'qwen3omni', 'chordino', 'essentia', 'essentia_instrument', "
            f"'beatnet', 'music_cpu_pipeline', 'music_cpu_lite_pipeline', 'dasm'"
        )

    entry = model_map[model_type]
    model_cls = entry['cls']
    if model_name is None:
        model_name = entry['default_name']

    options = entry['options']
    if options:
        return ModelWorker.options(**options).remote(model_cls, model_name, model_path, **kwargs)
    else:
        return ModelWorker.remote(model_cls, model_name, model_path, **kwargs)


# 兼容旧接口：为特定模型类型提供便捷工厂函数
def create_chordino_worker(model_name: str = None, model_path: str = None, **kwargs):
    """创建 Chordino 模型 Worker"""
    return create_worker(model_type="chordino", model_name=model_name, model_path=model_path, **kwargs)


def create_beatnet_worker(model_name: str = None, model_path: str = None, **kwargs):
    """创建 BeatNet 模型 Worker"""
    return create_worker(model_type="beatnet", model_name=model_name, model_path=model_path, **kwargs)


def create_essentia_worker(model_name: str = None, model_path: str = None, **kwargs):
    """创建 Essentia 模型 Worker"""
    return create_worker(model_type="essentia", model_name=model_name, model_path=model_path, **kwargs)
