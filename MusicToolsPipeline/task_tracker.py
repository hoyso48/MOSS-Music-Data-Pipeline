import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TaskTracker:
    """
    使用单个 JSONL 文件追踪任务状态。

    文件包含两类记录：
    - `{"type": "meta", "total_tasks": N, "timestamp": ...}`
    - `{"task_id": X, "completed_at": ...}`

    只需顺序读取即可恢复所有信息。
    """

    def __init__(self, record_path: str):
        self.record_path = record_path
        os.makedirs(os.path.dirname(self.record_path) or ".", exist_ok=True)

        self._completed: Set[int] = set()
        self._allocated_pending: Set[int] = set()
        self._next_candidate: int = 0
        self.total_tasks: Optional[int] = None

        self._load_records()
        self._rebuild_cursor()

    # ------------------------------------------------------------------ #
    # 状态加载与持久化
    # ------------------------------------------------------------------ #
    def _load_records(self):
        if not os.path.exists(self.record_path):
            return

        loaded_completed = 0
        loaded_meta = 0

        with open(self.record_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                if isinstance(entry, dict) and entry.get("type") == "meta":
                    total = entry.get("total_tasks")
                    if total is not None:
                        self.total_tasks = int(total)
                        loaded_meta += 1
                    continue

                task_id = entry.get("task_id") if isinstance(entry, dict) else None
                if task_id is not None:
                    self._completed.add(int(task_id))
                    loaded_completed += 1

        if loaded_completed:
            logger.info(f"Loaded {loaded_completed} completed tasks from {self.record_path}")
        if loaded_meta:
            logger.info(f"Loaded {loaded_meta} meta entries (total_tasks={self.total_tasks}) from {self.record_path}")

    def _append_records(self, records: List[Dict[str, Any]]):
        if not records:
            return
        with open(self.record_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    # 与旧接口兼容的方法
    # ------------------------------------------------------------------ #
    def init_tasks(self, total_tasks: int):
        """记录任务总数，便于统计；不再需要逐条初始化。"""
        self.total_tasks = int(total_tasks)
        self._append_records([{
            "type": "meta",
            "total_tasks": self.total_tasks,
            "timestamp": time.time()
        }])
        logger.info(f"Recorded total_tasks={total_tasks} in {self.record_path}")

    def reset_incomplete_allocations(self):
        """清空 pending 集合，确保重启后可以重新分配。"""
        if self._allocated_pending:
            logger.info(f"Clearing {len(self._allocated_pending)} pending allocations")
        self._allocated_pending.clear()
        self._rebuild_cursor()

    def _rebuild_cursor(self):
        """将游标移到首个未完成任务。"""
        if self.total_tasks is None:
            self._next_candidate = 0
            return
        idx = 0
        while idx < self.total_tasks and idx in self._completed:
            idx += 1
        self._next_candidate = idx

    def mark_tasks_completed(self, task_ids: List[int]):
        """批量追加 task_id 到 JSONL，并更新内存集合。"""
        if not task_ids:
            return

        new_ids = []
        for tid in task_ids:
            if tid is None:
                continue
            tid = int(tid)
            self._allocated_pending.discard(tid)
            if tid not in self._completed:
                new_ids.append(tid)
        if not new_ids:
            return

        timestamp = time.time()
        records = [{"task_id": tid, "completed_at": timestamp} for tid in new_ids]
        try:
            self._append_records(records)
            self._completed.update(new_ids)
            logger.debug(f"Appended {len(new_ids)} completed tasks to {self.record_path}")
        except Exception as e:
            logger.error(f"Failed to append completed tasks: {e}")
            raise
        self._rebuild_cursor()

    def get_completed_tasks(self) -> Set[int]:
        """返回已完成任务 ID 的集合副本。"""
        return set(self._completed)

    def mark_tasks_allocated(self, task_ids: List[int], worker_id: str = ""):
        """记录已分配但未完成的任务，防止重复发放。"""
        for tid in task_ids or []:
            if tid is None:
                continue
            tid = int(tid)
            if tid not in self._completed:
                self._allocated_pending.add(tid)

    def get_unallocated_tasks(self, batch_size: int) -> List[int]:
        """
        返回一批尚未完成且未被分配的任务 ID。
        顺序扫描即可满足 DescriptorLoader 的使用场景。
        """
        if self.total_tasks is None or self.total_tasks <= 0:
            logger.warning("Total tasks unknown, cannot allocate tasks deterministically")
            return []

        result: List[int] = []
        total = self.total_tasks
        idx = self._next_candidate
        checked = 0
        wrapped = False

        while len(result) < batch_size and checked < total:
            if idx >= total:
                if wrapped:
                    break
                idx = 0
                wrapped = True
                continue

            if idx not in self._completed and idx not in self._allocated_pending:
                result.append(idx)
            idx += 1
            checked += 1

        self._next_candidate = idx if idx < total else 0
        return result

    def get_progress_stats(self) -> Dict[str, int]:
        completed = len(self._completed)
        allocated = len(self._allocated_pending)
        total = self.total_tasks or (completed + allocated)
        unallocated = max(total - completed - allocated, 0)
        return {
            "completed": completed,
            "allocated": allocated,
            "unallocated": unallocated,
        }

    def is_complete(self) -> bool:
        if not self.total_tasks:
            return False
        return len(self._completed) >= self.total_tasks

    def cleanup(self):
        """兼容旧接口，无需额外清理。"""
        pass


# 兼容性函数，保持与原有代码的调用方式一致 ---------------------------
def init_task_tracking(db_path: str, total_tasks: int):
    tracker = TaskTracker(db_path)
    tracker.init_tasks(total_tasks)


def mark_tasks_completed(db_path: str, task_ids: List[int]):
    tracker = TaskTracker(db_path)
    tracker.mark_tasks_completed(task_ids)


def get_completed_tasks(db_path: str) -> Set[int]:
    tracker = TaskTracker(db_path)
    return tracker.get_completed_tasks()


def get_progress_stats(db_path: str) -> Dict[str, int]:
    tracker = TaskTracker(db_path)
    return tracker.get_progress_stats()
