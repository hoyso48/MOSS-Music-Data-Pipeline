# -*- coding: utf-8 -*-
"""
模型模块
提供统一的模型接口和实现
"""

from .base_model import BaseModel
from .beats_model import BEATsModel
# from .qwen3omni_model import Qwen3OmniModel
from .chordino_model import ChordinoModel
from .essentia_model import EssentiaModel
from .beatnet_model import BeatNetModel
from .pipeline_model import MusicCpuPipelineModel, MusicCpuLitePipelineModel
from .essentia_instrument_model import EssentiaInstrumentModel
from .dasm_model import DASMModel

__all__ = [
    'BaseModel',
    'BEATsModel',
    # 'Qwen3OmniModel',
    'ChordinoModel',
    'EssentiaModel',
    'EssentiaInstrumentModel',
    'BeatNetModel',
    'MusicCpuPipelineModel',
    'MusicCpuLitePipelineModel',
    'DASMModel',
]

