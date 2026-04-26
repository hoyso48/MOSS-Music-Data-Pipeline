# -*- coding: utf-8 -*-
"""
DASM (Detect Any Sound Model) 模型
用于检测音频中的特定声音
"""
import os
import sys
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import io

from .base_model import BaseModel
from audio_info import AudioInfo

logger = logging.getLogger(__name__)


class DASMModel(BaseModel):
    """DASM 声音检测模型"""

    def _load_model(self):
        """加载 DASM 和 MGA-CLAP 模型"""
        try:
            # 从配置获取路径（如果未提供，使用默认路径）
            root_path = self.config.get('root_path', 
                '/inspire/hdd/project/embodied-multimodality/public/audio_understanding/personal/zyfei/data_annotation/Transformer4SED')
            
            # 如果提供了完整路径，直接使用；否则基于 root_path 构建
            clap_weight_path = self.config.get('clap_weight_path')
            if clap_weight_path is None:
                clap_weight_path = os.path.join(root_path, 'recipes/audioset_strong/detect_any_sound/MGA-CLAP/mga-clap.pt')
            
            dasm_weight_path = self.config.get('dasm_weight_path')
            if dasm_weight_path is None:
                dasm_weight_path = os.path.join(root_path, 'docs/DASM/pretrained_model/detect_any_sound/text_query/as_full_text_query_best_model.pt')
            
            dasm_config_path = self.config.get('dasm_config_path')
            if dasm_config_path is None:
                dasm_config_path = os.path.join(root_path, 'docs/DASM/pretrained_model/detect_any_sound/text_query/config.yaml')
            
            # 默认查询列表
            self.query_list = self.config.get('query_list', 
                ['alarm', 'sirens', 'air raid sirens', 'dog', 'music', 'bird', 'speech', 'cat', 'inhale'])
            
            # 检测阈值
            self.threshold = self.config.get('threshold', 0.3)
            
            # 设备
            self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            
            logger.info(f"Loading DASM model on device: {self.device}")
            logger.info(f"Query list: {self.query_list}")
            
            # ========== 加载 MGA-CLAP 模型 ==========
            clap_dir = os.path.join(root_path, "recipes/audioset_strong/detect_any_sound/MGA-CLAP")
            if clap_dir not in sys.path:
                sys.path.insert(0, clap_dir)

            # 让 MGA-CLAP 下的 models 目录也出现在 sys.path 中，
            # 这样可以在 ase_model.py 中使用 `from audio_encoder import AudioEncoder`
            # 等同目录导入方式，避免顶层 `models` 包名冲突。
            mga_models_dir = os.path.join(clap_dir, "models")
            if os.path.isdir(mga_models_dir) and mga_models_dir not in sys.path:
                sys.path.insert(0, mga_models_dir)
            
            original_dir = os.getcwd()
            os.chdir(clap_dir)
            
            import yaml
            from ase_model import ASE
            
            config_path = "settings/inference_sed.yaml"
            with open(config_path, "r") as f:
                clap_config = yaml.safe_load(f)
            
            self.clap = ASE(clap_config)
            self.clap = self.clap.to(self.device)
            
            # 加载 CLAP 权重
            import numpy
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
            self.clap.load_state_dict(torch.load(clap_weight_path, weights_only=False)['model'], strict=False)
            self.clap.eval()
            
            os.chdir(original_dir)
            logger.info("MGA-CLAP model loaded")
            
            # ========== 加载 DASM 模型 ==========
            if root_path not in sys.path:
                sys.path.insert(0, root_path)
            
            os.chdir(root_path)
            
            from src.utils import load_yaml_with_relative_ref
            from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT
            
            configs = load_yaml_with_relative_ref(dasm_config_path)
            self.dasm_model = DASM_HTSAT(**configs["DASM_HTSAT"]["init_kwargs"])
            self.dasm_model.load_state_dict(torch.load(dasm_weight_path))
            self.dasm_model = self.dasm_model.to(self.device)
            self.dasm_model.eval()
            
            os.chdir(original_dir)
            logger.info("DASM model loaded")
            
            # ========== 初始化编码器 ==========
            from src.preprocess.feats_extraction import waveform_modification
            from src.codec.encoder import Encoder
            
            self.encoder = Encoder(
                [],
                audio_len=configs["feature"]["audio_max_len"],
                frame_len=configs["feature"]["win_length"],
                frame_hop=configs["feature"]["hopsize"],
                net_pooling=configs["feature"]["net_subsample"],
                sr=configs["feature"]["sr"],
            )
            
            self.waveform_modification = waveform_modification
            self.sample_rate = configs["feature"]["sr"]
            
            # ========== 预生成查询向量 ==========
            self._prepare_queries()
            
            logger.info("DASM model initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to load DASM model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _prepare_queries(self):
        """预生成查询向量"""
        import torch.nn.functional as F
        import torch.nn as nn
        
        # 生成查询向量
        prompt = 'sound of '
        queries = [prompt + x.lower() for x in self.query_list]
        
        with torch.no_grad():
            _, word_embeds, attn_mask = self.clap.encode_text(queries)
            text_embeds = self.clap.msc(word_embeds, self.clap.codebook, attn_mask)
            text_embeds = F.normalize(text_embeds, dim=-1)
        
        # 加载基础查询向量
        if not isinstance(self.dasm_model.at_query, nn.ParameterList):
            base_vector = self.dasm_model.at_query
        else:
            base_vector = self.dasm_model.at_query[0]  # text query
        
        self.base_size = len(base_vector)
        self.query_vectors = text_embeds
        self.base_vector = base_vector
        
        # 组合查询
        query = torch.cat([base_vector, text_embeds]).to(self.device)
        self.query = query
        
        # 生成注意力掩码
        query_len = query.shape[0]
        att_mask = torch.ones(query_len, query_len, dtype=torch.bool).to(self.device)
        att_mask[:, :self.base_size] = False
        att_mask.fill_diagonal_(False)
        self.att_mask = att_mask

    def _load_audio(self, audio_info: AudioInfo):
        """加载音频数据，返回 (wav, pad_mask) 元组"""
        # 优先使用 audio_bytes（Lance 等内存数据）
        if audio_info.audio_bytes is not None and isinstance(audio_info.audio_bytes, bytes):
            try:
                wav, pad_mask = self.waveform_modification(
                    audio_info.audio_bytes,
                    self.encoder.audio_len * self.encoder.sr,
                    self.encoder
                )
                return wav, pad_mask
            except Exception as e:
                logger.warning(f"Failed to decode audio_bytes: {e}")
                return None, None
        
        # 退回到使用文件路径 / URL
        if audio_info.audio_path or audio_info.url or audio_info.path:
            audio_path = audio_info.audio_path or audio_info.url or audio_info.path
            try:
                wav, pad_mask = self.waveform_modification(
                    audio_path, 
                    self.encoder.audio_len * self.encoder.sr, 
                    self.encoder
                )
                return wav, pad_mask
            except Exception as e:
                logger.warning(f"Failed to load audio from file {audio_path}: {e}")
                return None, None
        else:
            logger.warning("No audio_path or audio_bytes in input")
            return None, None

    def generate(self, inputs: List[AudioInfo], **kwargs) -> List[AudioInfo]:
        """处理 AudioInfo 列表，返回检测结果"""
        results: List[AudioInfo] = []
        
        for audio_info in inputs:
            if audio_info._extra is None:
                audio_info._extra = {}
            
            try:
                # 加载音频
                wav, pad_mask = self._load_audio(audio_info)
                if wav is None:
                    audio_info.error = "Failed to load audio"
                    results.append(audio_info)
                    continue
                
                # 添加 batch 维度并移到设备
                wav = wav.unsqueeze(0).to(self.device)
                pad_mask = pad_mask.unsqueeze(0).to(self.device)
                
                # 提取 mel 频谱特征
                extractor = self.dasm_model.get_feature_extractor()
                mel = extractor(wav)
                
                # 运行检测
                with torch.no_grad():
                    strong, weak, other_dict = self.dasm_model(
                        input=mel,
                        temp_w=0.5,
                        pad_mask=pad_mask,
                        query=self.query,
                        query_type='text',
                        tgt_mask=self.att_mask,
                    )
                
                # 提取结果
                at_scores = other_dict['at_out'].squeeze(0)[self.base_size:].detach().cpu().numpy()
                strong_scores = strong.squeeze(0)[self.base_size:].detach().cpu().numpy()
                
                # 仅保留每个 label 的整体预测分数
                detections = []
                for i, sound_name in enumerate(self.query_list):
                    at_score = float(at_scores[i])
                    detections.append({
                        'sound': sound_name,
                        'score': at_score,
                    })
                
                audio_info._extra['dasm'] = detections
                
            except Exception as e:
                logger.error(f"Error processing audio {audio_info.get_audio_identifier()}: {e}")
                audio_info.error = str(e)
            
            results.append(audio_info)
        
        return results

    def generate_batch(self, batch_data: List[AudioInfo]) -> List[AudioInfo]:
        """处理一批 AudioInfo 数据"""
        normalized_inputs: List[AudioInfo] = []
        for item in batch_data:
            if isinstance(item, AudioInfo):
                audio_info = item
            elif isinstance(item, dict):
                audio_info = AudioInfo.from_dict(item)
            else:
                audio_info = AudioInfo(audio_path=str(item))
            if audio_info._extra is None:
                audio_info._extra = {}
            normalized_inputs.append(audio_info)
        return self.generate(normalized_inputs)

    def cleanup(self):
        """清理模型资源"""
        if hasattr(self, 'dasm_model'):
            del self.dasm_model
        if hasattr(self, 'clap'):
            del self.clap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
