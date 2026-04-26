# *Ray Inference Framework*

*一个基于 [Ray](https://www.ray.io/) 的分布式推理框架，用于高效处理大规模音频数据。框架采用队列式流水线架构，支持多 GPU 并行推理、断点续跑和灵活的数据源接入。*

## *特性*

- *🚀 **分布式推理**：基于 Ray 实现多 GPU 并行处理*
- *📦 **队列式流水线**：解耦数据加载、模型推理和结果保存*
- *🔄 **断点续跑**：基于 SQLite 的任务跟踪，支持中断后恢复*
- *🔌 **灵活扩展**：易于添加新模型和数据加载器*
- *📊 **统一数据格式**：使用 `AudioInfo` 类统一数据流*

## *支持的模型*

*- **BEATsModel**：开源的通用音频表征模型，用于音频分类与检索*

*- **Qwen3OmniModel**：基于 vLLM 的多模态生成模型，支持语音/文本对话*

*- **ChordinoModel**：Chordino 谱和弦分析模型，输出和弦序列*

*- **EssentiaModel**：Essentia 音频分析模型，适用于通用音频属性提取*

*- **EssentiaInstrumentModel**：Essentia 乐器识别模型，输出乐器置信度*

*- **BeatNetModel**：BeatNet 节拍/下拍检测模型*

*- **MusicCpuPipelineModel**：MUSIC CPU模型，也即 Chordino+Essentia+BeatNet*

## *支持的数据加载器*

*-**AudioJSONLDataLoader**：从本地 JSONL 文件加载音频路径*

*-**OSSDataloader**：从 OSS（对象存储）加载音频数据，支持预签名 URL 和 FFmpeg 解码*

*-**JSONLDataLoader**：通用 JSONL 数据加载器*

*-**LanceDataLoader**：面向 Lance 列式数据集的加载器，支持本地/S3 数据、断点续跑以及列裁剪*

## *整体流程*

*框架采用队列式流水线架构，数据在各组件间通过 Ray Queue 传递：*

```

┌─────────────────┐

│ DataLoaderWorker │  顺序迭代数据，跳过已完成任务

└────────┬────────┘

         │ AudioInfo[]

         ▼

┌─────────────────┐

│   input_queue    │  输入队列（Ray Queue）

└────────┬────────┘

         │ AudioInfo[]

         ▼

┌─────────────────┐

│  ModelWorker(s) │  并行处理，支持多 GPU / CPU

│  (可动态扩展)    │

└────────┬────────┘

         │ AudioInfo[] (含预测结果)

         ▼

┌─────────────────┐

│  result_queue   │  结果队列（Ray Queue）

└────────┬────────┘

         │ AudioInfo[]

         ▼

┌─────────────────┐

│   SaveWorker    │  批量保存结果，更新任务状态

└────────┬────────┘

         │

         ▼

┌─────────────────┐

│  TaskTracker    │  SQLite 数据库（任务状态跟踪）

│  (SaveWorker    │  直接调用 mark_tasks_completed

│   内部使用)      │

└─────────────────┘

```

### *工作流程说明*

*1. **DataLoaderWorker**：在 Ray Actor 中构建数据加载器，通过 `TaskTracker` 查询已完成任务并在 Lance 模式下应用 offset/limit 分片；顺序迭代 `AudioInfo`，对内存中的音频 bytes 使用 `ray.put` 共享引用，将批次与索引放入 `input_queue` 并在收尾阶段发送结束信号*

*2.**ModelWorker(s)**：从 `input_queue` 获取批次数据，调用模型进行推理，将结果放入 `result_queue`*

*3.**SaveWorker**：从 `result_queue` 获取结果，批量写入文件并更新数据库中的任务状态*

*4.**TaskTracker**：使用 SQLite 跟踪任务状态（unallocated → completed），支持断点续跑*

## *快速开始*

### *安装依赖*

```bash
# 基础依赖
pip install -U ray torch torchaudio soundfile

# LanceDataLoader 依赖（可选）
pip install -U lance pyarrow daft

# OSSDataloader 依赖（可选）
pip install -U minio ffmpeg-python

# Qwen3-Omni 依赖（可选）
pip install -U vllm
```

### *基本使用*

```bash
# 模型权重和数据路径由命令行参数传入
python ray_inference.py \
  --model /path/to/model_dir_or_ckpt \
  --data-path s3://bucket/path/to/data.lance \
  --output ./outputs 
```

### *配置修改*

*所有配置都在 `config.py` 中的 `Config` 类中管理。修改 `ray_inference.py` 的 `__main__` 部分：*

```python
# 数据路径
cfg.data_path = '/path/to/data.lance'  # 或 JSONL/OSS 目录
cfg.output_path = './outputs'

# 模型和数据加载器
cfg.model_type = 'essentia_instrument'  # beats, qwen3omni, chordino, essentia, beatnet, music_cpu_pipeline
cfg.dataloader_type = 'lance'           # oss, audio_jsonl, jsonl, lance

# Lance 配置（仅 LanceDataLoader 使用）
cfg.lance_prompt_key = 'audio_flac'
cfg.lance_offset = 0
cfg.lance_limit = None

# OSS 配置（仅 OSSDataloader 使用）
cfg.meta_dir = '/path/to/metadata'
cfg.endpoint = 'oss.example.com:8009'
cfg.access_key = 'your_access_key'
cfg.secret_key = 'your_secret_key'

# 推理配置
cfg.batch_size = 4
cfg.group_by_segment = True
```

## *caption pipeline的启动方法*

### *beats*

- 在`ray_inference.py`中设置好核心参数：`model_type=beats`、`dataloader_type=lance/jsonl`、`cfg.batch_size=128`（视显存和速度调节）、`cfg.lance_prompt_key="audio_flac"`(选填，默认使用lance，设置好type为jsonl则忽略)。

- 启动示例：
  ```bash
  MODEL_TYPE=beats DATALOADER_TYPE=lance \
  python ray_inference.py \
    --cfg model_path=/xxx/xxx/ckpts/BEATs_iter3_plus_AS2M_finetuned.pt \
    --cfg data_path=xxxx.jsonl \
    --cfg model_type=beats  \
    --cfg gpu_per_worker=0.05 \
    --cfg batch_size=128   \
    --cfg dataloader_type=jsonl \
    --cfg output_path=./work_dir/beats_debug/
  ```
- 说明：`--model` 需要指向 BEATs checkpoint；
- 可以参考`ray_inference.sh`

### *music cpu*

- 适用说明：需要并行抽取和弦（Chordino）、节拍（BeatNet）和旋律/调式（Essentia）的多任务 caption pipeline，全 CPU 执行，适合 CPU 资源充足的空间，比如高性能计算区，CPU资源空间。

- 核心参数：`model_type=music_cpu_pipeline`、`dataloader_type=lance`，以及 `cfg.batch_size=128`。如需微调每个子 worker 的 CPU 份额，可在 `workers/model_worker.py` 的 `create_music_cpu_pipeline_worker` 中传入 `chordino_num_cpus`/`beatnet_num_cpus`/`essentia_num_cpus`，或在 `MusicCpuPipelineModel` 内根据情况调整。

- 启动示例（Pipeline 不依赖单一 checkpoint，但仍需占位 `--model`,随便写一个即可）：
  ```bash
  python ray_inference.py \
  --cfg data_path="${DATA_PATH}" \
  --cfg model_path=dummy \
  --cfg output_path="${OUTPUT_PATH}" \
  --cfg model_type=music_cpu_pipeline \
  --cfg num_workers=35 \
  --cfg batch_size=4 \
  --cfg num_dataloader_workers=4 \
  ```
- 结果会自动写入 `["music_cpu"]`，包含 `chords`/`beatnet`/`melody`/`key` 等字段，方便后续 caption 逻辑直接消费。
- 也可以直接参考脚本`script/music_cpu/start.sh`

### *music gpu*
- 同上，与music cpu相似，不过要设置`model_type=essentia_instrument`.
- 也可以直接参考脚本`script/music_gpu/start.sh`
## *扩展指南*

### *添加新模型*

*在 `models/` 目录下创建新模型文件，继承 `BaseModel` 类：至少实现 `_load_model()`（初始化权重或客户端）与 `generate()`（处理输入批次并写入 `predictions`），如需自定义批处理可重写 `generate_batch()`：*

```python
from models.base_model import BaseModel
from audio_info import AudioInfo
from typing import List


class MyModel(BaseModel):
    def __init__(self, model_name: str, model_path: str = None, **kwargs):
        super().__init__(model_name, model_path, **kwargs)
        self._load_model()

    def _load_model(self):
        """加载模型（权重、adapter、客户端等）"""
        # TODO: implement model loading logic
        ...

    def generate(self, inputs: List[AudioInfo], **kwargs) -> List[AudioInfo]:
        """处理 AudioInfo 列表，写入 predictions 后返回"""
        results = []
        for audio_info in inputs:
            audio_info.predictions = [...]  # your predictions
            results.append(audio_info)
        return results

    def generate_batch(self, batch_data: List[AudioInfo], **kwargs) -> List[AudioInfo]:
        """可选：覆盖默认实现以优化批处理"""
        return self.generate(batch_data, **kwargs)
```

*然后在 `workers/model_worker.py` 中添加创建函数：*

```python
def create_my_model_worker(model_name: str, model_path: str = None, **kwargs):
    from models.my_model import MyModel
    return ModelWorker.remote(MyModel, model_name, model_path, **kwargs)
```

### *添加新数据加载器*

*在 `dataloader.py` 中创建新数据加载器，继承 `BaseDataLoader` 类：*

```python
from dataloader import BaseDataLoader
from audio_info import AudioInfo
from typing import Iterator, List


class MyDataLoader(BaseDataLoader):
    def _load_data(self):
        """加载数据到 self.data"""
        self.data = [...]  # 数据列表（字典格式）

    def __iter__(self) -> Iterator[List[AudioInfo]]:
        """返回批次数据"""
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i + self.batch_size]
            yield [AudioInfo.from_dict(item) for item in batch]

    def get_item(self, index: int) -> dict:
        """根据索引获取单个数据项"""
        return self.data[index]
```

*然后在 `dataloader.py` 的 `create_dataloader` 函数中添加：*

```python
def create_dataloader(dataloader_type: str = 'oss', ...):
    # ...
    elif dataloader_type == 'my_dataloader':
        return MyDataLoader(data_path=data_path, batch_size=batch_size, ...)
```

#### *数据格式参考*

*扩展模型或数据加载器时，记得复用统一的 `AudioInfo` 数据结构：*

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class AudioInfo:
    # 音频路径/URL
    audio_path: Optional[str] = None
    url: Optional[str] = None

    # OSS 相关参数（用于 OSSDataloader）
    bucket: Optional[str] = None
    object_key: Optional[str] = None
    endpoint: Optional[str] = None

    # Lance/OSS 字段
    audio_bytes: Optional[bytes] = None
    _id: Optional[str] = None
    metadata_oid: Optional[str] = None
    segment_key: Optional[str] = None

    # 音频片段信息
    start: Optional[float] = None
    end: Optional[float] = None

    # 模型输出
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    # 额外字段
    _extra: Dict[str, Any] = field(default_factory=dict)
```

## *架构设计*

### *组件说明*

*-**DataLoaderWorker**：数据加载工作器，负责从数据源加载数据并放入队列*

*-**ModelWorker**：模型推理工作器，每个 GPU 一个实例，从队列获取数据并推理*

*-**SaveWorker**：结果保存工作器，批量保存结果并更新任务状态*

*-**QueueMonitor**：队列监控器，定期输出队列大小信息*

*-**TaskTracker**：任务跟踪器，使用 SQLite 管理任务状态*

### *断点续跑机制*

*框架使用 SQLite 数据库跟踪任务状态：*

- *启动时查询已完成的任务，跳过这些任务*
- *处理完成后标记任务为 `completed`*
- *支持中断后恢复，自动跳过已完成的任务*

### *性能优化*

*-**批量处理**：SaveWorker 使用缓冲区批量写入文件和数据库*

*-**异步流水线**：数据加载、推理和保存并行执行*

*-**资源管理**：支持多 worker 共享 GPU*

## *示例*

*完整示例请参考 `example_ray_run.py`。*

## *大规模分布式处理建议*

- 对于需要批量起任务的场景，`ray_inference.py` 暴露了 `--worker-rank` 与 `--world-size` 参数，单次启动即可在多节点间进行简单的数据分片。
- 在较大规模的数据语料上，我们更推荐以“单节点任务 + 外部调度器批量提交”的方式，通过你自己的 HPC / Kubernetes / Ray 集群批量提交单节点作业，既能按需控制任务优先级、最大化资源利用，也能避免分片粒度过粗造成的数据倾斜（部分节点数据过多、其他节点空等）以及调度浪费。
- 这种方式还有一个好处：某个节点失败时不会波及其他节点，重试与排障更高效。


## *贡献*

*欢迎提交 Issue 和 Pull Request！*
