#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import time
from argparse import Namespace
from pathlib import Path
import sys
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "third_party"))

# monkey patch to fix issues in msaf
import scipy
import numpy as np

scipy.inf = np.inf

import librosa
import torch
from ema_pytorch import EMA
from loguru import logger
from muq import MuQ
from musicfm.model.musicfm_25hz import MusicFM25Hz
from omegaconf import OmegaConf
from tqdm import tqdm

mp.set_start_method("spawn", force=True)

MUSICFM_HOME_PATH = os.path.join("ckpts", "MusicFM")

BEFORE_DOWNSAMPLING_FRAME_RATES = 25
AFTER_DOWNSAMPLING_FRAME_RATES = 8.333

DATASET_LABEL = "SongForm-HX-8Class"
DATASET_IDS = [5]

TIME_DUR = 420
INPUT_SAMPLING_RATE = 24000

from dataset.label2id import DATASET_ID_ALLOWED_LABEL_IDS, DATASET_LABEL_TO_DATASET_ID
from postprocessing.functional import postprocess_functional_structure


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def uid_from_line_no(line_no: int) -> str:
    return f"{line_no:09d}"


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def load_checkpoint(checkpoint_path, device=None):
    """Load checkpoint from path"""
    if device is None:
        device = "cpu"

    if checkpoint_path.endswith(".pt"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
    elif checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        checkpoint = {"model_ema": load_file(checkpoint_path, device=device)}
    else:
        raise ValueError("Unsupported checkpoint format. Use .pt or .safetensors")
    return checkpoint


def rule_post_processing(msa_list):
    if len(msa_list) <= 2:
        return msa_list

    result = msa_list.copy()

    while len(result) > 2:
        first_duration = result[1][0] - result[0][0]
        if first_duration < 1.0 and len(result) > 2:
            result[0] = (result[0][0], result[1][1])
            result = [result[0]] + result[2:]
        else:
            break

    while len(result) > 2:
        last_label_duration = result[-1][0] - result[-2][0]
        if last_label_duration < 1.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    while len(result) > 2:
        if result[0][1] == result[1][1] and result[1][0] <= 10.0:
            result = [(result[0][0], result[0][1])] + result[2:]
        else:
            break

    while len(result) > 2:
        last_duration = result[-1][0] - result[-2][0]
        if result[-2][1] == result[-3][1] and last_duration <= 10.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    return result


def get_processed_uids(output_dir: str):
    """uids that already have uid.json in output_dir"""
    p = Path(output_dir)
    if not p.exists():
        return set()
    ret = set()
    for x in p.iterdir():
        if x.is_file() and x.suffix == ".json":
            ret.add(x.stem)
    return ret


def build_tasks_from_jsonl(input_jsonl: str, audio_key: str, processed_uids: set):
    """
    Return list of tasks: (uid, audio_path)
    uid is 9-digit line number, unique & stable.
    """
    tasks = []
    total = count_lines(input_jsonl)  # 先扫一遍统计总行数（更准的进度/ETA）

    for line_no, obj in tqdm(
        iter_jsonl(input_jsonl),
        total=total,
        desc="scan jsonl -> tasks",
        unit="lines",
        dynamic_ncols=True,
    ):
        uid = uid_from_line_no(line_no)
        if uid in processed_uids:
            continue
        audio_path = obj.get(audio_key, None)
        if not audio_path:
            # still keep progress stable, but skip
            logger.warning(f"missing key '{audio_key}' at line {line_no}, skip")
            continue
        tasks.append((uid, audio_path))

    return tasks


def inference_worker(rank, queue_input: mp.Queue, queue_output: mp.Queue, args: Namespace):
    """
    Each task is (uid, audio_path).
    Writes output to output_dir/uid.json
    Always puts one message into queue_output per task:
      {"uid": uid, "ok": True} or {"uid": uid, "ok": False, "error": "..."}
    """
    device = f"cuda:{rank}"

    # reduce CPU thread contention when many processes
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # MuQ model loading (auto fetch)
    muq = MuQ.from_pretrained("ckpts/MuQ-large-msd-iter")
    muq = muq.to(device).eval()

    # MusicFM loading
    musicfm = MusicFM25Hz(
        is_flash=False,
        stat_path=os.path.join(MUSICFM_HOME_PATH, "msd_stats.json"),
        model_path=os.path.join(MUSICFM_HOME_PATH, "pretrained_msd.pt"),
    )
    musicfm = musicfm.to(device).eval()

    # Custom model loading
    module = importlib.import_module("models." + str(args.model))
    Model = getattr(module, "Model")
    hp = OmegaConf.load(os.path.join("configs", args.config_path))
    model = Model(hp)

    ckpt = load_checkpoint(checkpoint_path=os.path.join("ckpts", args.checkpoint), device="cpu")
    if ckpt.get("model_ema", None) is not None:
        logger.info(f"[rank {rank}] Loading EMA model parameters")
        model_ema = EMA(model, include_online_model=False)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(model_ema.ema_model.state_dict())
    else:
        logger.info(f"[rank {rank}] No EMA model parameters found, using original model")
        model.load_state_dict(ckpt["model"])

    model.to(device).eval()

    num_classes = args.num_classes
    dataset_id2label_mask = {}
    for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
        dataset_id2label_mask[key] = np.ones(args.num_classes, dtype=bool)
        dataset_id2label_mask[key][allowed_ids] = False

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        while True:
            item = queue_input.get()
            if item is None:
                break

            uid, audio_path = item
            out_path = os.path.join(args.output_dir, f"{uid}.json")

            # resume-safe: if exists, skip compute
            if os.path.exists(out_path):
                queue_output.put({"uid": uid, "ok": True, "skipped": True})
                continue

            try:
                wav, sr = librosa.load(audio_path, sr=INPUT_SAMPLING_RATE)
                audio = torch.tensor(wav).to(device)

                win_size = args.win_size
                hop_size = args.hop_size

                total_len = ((audio.shape[0] // INPUT_SAMPLING_RATE) // TIME_DUR) * TIME_DUR + TIME_DUR
                total_frames = math.ceil(total_len * AFTER_DOWNSAMPLING_FRAME_RATES)

                logits = {
                    "function_logits": np.zeros([total_frames, num_classes], dtype=np.float32),
                    "boundary_logits": np.zeros([total_frames], dtype=np.float32),
                }
                logits_num = {
                    "function_logits": np.zeros([total_frames, num_classes], dtype=np.float32),
                    "boundary_logits": np.zeros([total_frames], dtype=np.float32),
                }

                lens = 0
                i = 0
                while True:
                    start_idx = i * INPUT_SAMPLING_RATE
                    end_idx = min((i + win_size) * INPUT_SAMPLING_RATE, audio.shape[-1])
                    if start_idx >= audio.shape[-1]:
                        break
                    if end_idx - start_idx <= 1024:
                        i += hop_size
                        continue

                    audio_seg = audio[start_idx:end_idx]

                    # MuQ embedding (420s)
                    muq_output = muq(audio_seg.unsqueeze(0), output_hidden_states=True)
                    muq_embd_420s = muq_output["hidden_states"][10]
                    del muq_output
                    torch.cuda.empty_cache()

                    # MusicFM embedding (420s)
                    _, musicfm_hidden_states = musicfm.get_predictions(audio_seg.unsqueeze(0))
                    musicfm_embd_420s = musicfm_hidden_states[10]
                    del musicfm_hidden_states
                    torch.cuda.empty_cache()

                    # wrap 30s embeddings inside this hop window
                    wraped_muq_embd_30s = []
                    wraped_musicfm_embd_30s = []

                    for idx_30s in range(i, i + hop_size, 30):
                        start_idx_30s = idx_30s * INPUT_SAMPLING_RATE
                        end_idx_30s = min(
                            (idx_30s + 30) * INPUT_SAMPLING_RATE,
                            audio.shape[-1],
                            (i + hop_size) * INPUT_SAMPLING_RATE,
                        )
                        if start_idx_30s >= audio.shape[-1]:
                            break
                        if end_idx_30s - start_idx_30s <= 1024:
                            continue

                        wraped_muq_embd_30s.append(
                            muq(audio[start_idx_30s:end_idx_30s].unsqueeze(0), output_hidden_states=True)["hidden_states"][10]
                        )
                        torch.cuda.empty_cache()

                        wraped_musicfm_embd_30s.append(
                            musicfm.get_predictions(audio[start_idx_30s:end_idx_30s].unsqueeze(0))[1][10]
                        )
                        torch.cuda.empty_cache()

                    if len(wraped_muq_embd_30s) == 0 or len(wraped_musicfm_embd_30s) == 0:
                        i += hop_size
                        continue

                    wraped_muq_embd_30s = torch.concatenate(wraped_muq_embd_30s, dim=1)
                    wraped_musicfm_embd_30s = torch.concatenate(wraped_musicfm_embd_30s, dim=1)

                    all_embds = [
                        wraped_musicfm_embd_30s,
                        wraped_muq_embd_30s,
                        musicfm_embd_420s,
                        muq_embd_420s,
                    ]

                    # align lengths
                    if len(all_embds) > 1:
                        embd_lens = [x.shape[1] for x in all_embds]
                        max_embd_len = max(embd_lens)
                        min_embd_len = min(embd_lens)
                        if abs(max_embd_len - min_embd_len) > 4:
                            raise ValueError(f"Embedding shapes differ too much: {max_embd_len} vs {min_embd_len}")
                        for idx in range(len(all_embds)):
                            all_embds[idx] = all_embds[idx][:, :min_embd_len, :]

                    embd = torch.concatenate(all_embds, axis=-1)

                    dataset_label = DATASET_LABEL
                    dataset_ids = torch.Tensor(DATASET_IDS).to(device, dtype=torch.long)

                    msa_info, chunk_logits = model.infer(
                        input_embeddings=embd,
                        dataset_ids=dataset_ids,
                        label_id_masks=torch.Tensor(
                            dataset_id2label_mask[DATASET_LABEL_TO_DATASET_ID[dataset_label]]
                        ).to(device, dtype=bool).unsqueeze(0).unsqueeze(0),
                        with_logits=True,
                    )

                    start_frame = int(i * AFTER_DOWNSAMPLING_FRAME_RATES)
                    end_frame = start_frame + min(
                        math.ceil(hop_size * AFTER_DOWNSAMPLING_FRAME_RATES),
                        chunk_logits["boundary_logits"][0].shape[0],
                    )

                    logits["function_logits"][start_frame:end_frame, :] += chunk_logits["function_logits"][0].detach().cpu().numpy()
                    logits["boundary_logits"][start_frame:end_frame] = chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                    logits_num["function_logits"][start_frame:end_frame, :] += 1
                    logits_num["boundary_logits"][start_frame:end_frame] += 1
                    lens += end_frame - start_frame

                    i += hop_size

                # avoid divide-by-zero
                logits_num["function_logits"][logits_num["function_logits"] == 0] = 1
                logits_num["boundary_logits"][logits_num["boundary_logits"] == 0] = 1

                logits["function_logits"] /= logits_num["function_logits"]
                logits["boundary_logits"] /= logits_num["boundary_logits"]

                logits["function_logits"] = torch.from_numpy(logits["function_logits"][:lens]).unsqueeze(0)
                logits["boundary_logits"] = torch.from_numpy(logits["boundary_logits"][:lens]).unsqueeze(0)

                msa_infer_output = postprocess_functional_structure(logits, hp)

                # expected last token is end
                assert msa_infer_output[-1][-1] == "end"
                if not args.no_rule_post_processing:
                    msa_infer_output = rule_post_processing(msa_infer_output)

                msa_json = []
                for idx in range(len(msa_infer_output) - 1):
                    msa_json.append(
                        {
                            "label": msa_infer_output[idx][1],
                            "start": float(msa_infer_output[idx][0]),
                            "end": float(msa_infer_output[idx + 1][0]),
                        }
                    )

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(msa_json, f, indent=4, ensure_ascii=False)

                queue_output.put({"uid": uid, "ok": True})

            except Exception as e:
                queue_output.put({"uid": uid, "ok": False, "error": str(e)})
                logger.error(f"process {rank} error\nuid={uid}\naudio={audio_path}\n{e}")


def merge_back_to_jsonl(input_jsonl: str, output_dir: str, output_jsonl: str, audio_key: str, resume_merge: bool):
    """
    Write a new jsonl with added field: songformer_result
    If resume_merge and output_jsonl exists, append from its current line count.
    """
    start_line = 0
    mode = "w"
    if resume_merge and os.path.exists(output_jsonl):
        start_line = count_lines(output_jsonl)
        mode = "a"
        logger.info(f"resume_merge enabled, output_jsonl has {start_line} lines, will append from line {start_line}")

    with open(output_jsonl, mode, encoding="utf-8") as wf:
        pbar = tqdm(desc="merge back to jsonl", unit="lines")
        for line_no, obj in iter_jsonl(input_jsonl):
            if line_no < start_line:
                continue

            uid = uid_from_line_no(line_no)
            pred_path = os.path.join(output_dir, f"{uid}.json")
            if os.path.exists(pred_path):
                with open(pred_path, "r", encoding="utf-8") as pf:
                    obj["songformer_result"] = json.load(pf)
            else:
                obj["songformer_result"] = None

            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
            pbar.update(1)
        pbar.close()


def main():
    parser = argparse.ArgumentParser()

    # jsonl-only inputs
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input jsonl path")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output jsonl path (with songformer_result)")
    parser.add_argument("--audio_key", type=str, default="audio_path", help="audio path key in jsonl")

    # inference outputs
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to store per-item uid.json results")

    # parallelism
    parser.add_argument("--gpu_num", "-gn", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num_thread_per_gpu", "-tn", type=int, default=1, help="Processes per GPU")

    # model config
    parser.add_argument("--model", type=str, required=True, help="Model to use, e.g., SongFormer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint filename under ckpts/, e.g., SongFormer.safetensors")
    parser.add_argument("--config_path", type=str, required=True, help="Config filename under configs/, e.g., SongFormer.yaml")
    parser.add_argument("--num_classes", type=int, default=128)

    # behavior
    parser.add_argument("--no_rule_post_processing", action="store_true", help="Disable rule-based post-processing")
    parser.add_argument("--debug", action="store_true", help="Debug mode: run only 1 item on GPU 0")
    parser.add_argument("--resume_merge", action="store_true", help="Resume merging to output_jsonl (append)")
    parser.add_argument("--force_merge_only", action="store_true", help="Skip inference, only merge uid.json -> output_jsonl")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # merge-only mode
    if args.force_merge_only:
        merge_back_to_jsonl(args.input_jsonl, args.output_dir, args.output_jsonl, args.audio_key, args.resume_merge)
        return

    processed_uids = get_processed_uids(args.output_dir)
    tasks = build_tasks_from_jsonl(args.input_jsonl, args.audio_key, processed_uids)

    logger.info(f"output_dir: {args.output_dir}")
    logger.info(f"already processed: {len(processed_uids)}")
    logger.info(f"to process now: {len(tasks)}")

    init_args = Namespace(
        output_dir=args.output_dir,
        win_size=420,
        hop_size=420,
        num_classes=args.num_classes,
        model=args.model,
        checkpoint=args.checkpoint,
        config_path=args.config_path,
        no_rule_post_processing=args.no_rule_post_processing,
    )

    if args.debug:
        if len(tasks) == 0:
            logger.warning("no tasks to run (all processed). will still merge.")
        else:
            queue_input: mp.Queue = mp.Queue()
            queue_output: mp.Queue = mp.Queue()
            queue_input.put(tasks[0])
            queue_input.put(None)
            inference_worker(0, queue_input, queue_output, init_args)
            # consume one output message
            _ = queue_output.get()
        merge_back_to_jsonl(args.input_jsonl, args.output_dir, args.output_jsonl, args.audio_key, args.resume_merge)
        return

    gpu_num = args.gpu_num
    num_thread_per_gpu = args.num_thread_per_gpu
    num_workers = gpu_num * num_thread_per_gpu

    if num_workers <= 0:
        raise ValueError("gpu_num * num_thread_per_gpu must be > 0")

    queue_input: mp.Queue = mp.Queue(maxsize=2048)
    queue_output: mp.Queue = mp.Queue()

    processes = []
    for worker_idx in range(num_workers):
        rank = worker_idx % gpu_num
        logger.info(f"spawn worker {worker_idx} on GPU {rank}")
        time.sleep(0.05)
        p = mp.Process(
            target=inference_worker,
            args=(rank, queue_input, queue_output, init_args),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # enqueue tasks with progress
    for item in tqdm(tasks, desc="enqueue tasks", unit="item"):
        queue_input.put(item)

    # stop signals
    for _ in range(num_workers):
        queue_input.put(None)

    # collect outputs with progress
    ok = 0
    fail = 0
    skipped = 0
    pbar = tqdm(total=len(tasks), desc="inference done", unit="item")
    for _ in range(len(tasks)):
        msg = queue_output.get()
        if msg.get("ok", False):
            ok += 1
            if msg.get("skipped", False):
                skipped += 1
        else:
            fail += 1
        pbar.update(1)
        pbar.set_postfix({"ok": ok, "fail": fail, "skipped": skipped})
    pbar.close()

    for p in processes:
        p.join()

    # merge to output jsonl
    merge_back_to_jsonl(args.input_jsonl, args.output_dir, args.output_jsonl, args.audio_key, args.resume_merge)
    logger.info(f"all done. merged jsonl: {args.output_jsonl}")


if __name__ == "__main__":
    main()