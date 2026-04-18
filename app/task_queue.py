"""
Background Task Queue untuk TB Detector
Thread-based task queue untuk training dan preprocessing
"""

import threading
import queue
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass, asdict
import time
import traceback


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    PREDICTION = "prediction"
    EXPORT = "export"


@dataclass
class Task:
    """Task definition"""
    id: str
    type: TaskType
    name: str
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    params: Dict[str, Any] = None
    result: Any = None
    error: Optional[str] = None
    progress: int = 0
    current_step: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.name,
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'params': self.params,
            'result': self.result,
            'error': self.error,
            'progress': self.progress,
            'current_step': self.current_step
        }


class TaskQueue:
    """
    Thread-based task queue dengan priority support
    """
    
    def __init__(self, max_workers: int = 2, persistence=None):
        self.max_workers = max_workers
        self.persistence = persistence
        
        # Task queues (priority: lower number = higher priority)
        self._queues = {
            TaskType.PREDICTION: queue.PriorityQueue(),  # Highest priority
            TaskType.PREPROCESSING: queue.PriorityQueue(),
            TaskType.TRAINING: queue.PriorityQueue(),
            TaskType.EXPORT: queue.PriorityQueue(),
        }
        
        # Task storage
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        
        # Worker threads
        self._workers: List[threading.Thread] = []
        self._shutdown = threading.Event()
        self._current_tasks: Dict[int, str] = {}  # worker_id -> task_id
        
        # Callbacks
        self._on_task_complete: Optional[Callable[[Task], None]] = None
        self._on_task_fail: Optional[Callable[[Task, Exception], None]] = None
        self._on_progress: Optional[Callable[[str, int, str], None]] = None
        
        # Start workers
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self, worker_id: int):
        """Worker thread main loop"""
        while not self._shutdown.is_set():
            task = None
            
            # Try to get task dari queues (in priority order)
            for task_type in [TaskType.PREDICTION, TaskType.PREPROCESSING, TaskType.TRAINING, TaskType.EXPORT]:
                try:
                    # PriorityQueue returns (priority, task)
                    _, task = self._queues[task_type].get(timeout=0.1)
                    break
                except queue.Empty:
                    continue
            
            if task is None:
                continue
            
            # Execute task
            self._execute_task(task, worker_id)
    
    def _execute_task(self, task: Task, worker_id: int):
        """Execute a task"""
        with self._lock:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            self._current_tasks[worker_id] = task.id
        
        # Progress callback wrapper
        def progress_callback(progress: int, step: str):
            task.progress = progress
            task.current_step = step
            if self._on_progress:
                self._on_progress(task.id, progress, step)
        
        try:
            # Get handler untuk task type
            handler = self._get_task_handler(task.type)
            
            # Execute
            result = handler(task, progress_callback)
            
            # Mark complete
            with self._lock:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()
                task.result = result
                task.progress = 100
            
            # Persist jika ada persistence
            if self.persistence:
                self._persist_task_completion(task)
            
            # Callback
            if self._on_task_complete:
                self._on_task_complete(task)
            
        except Exception as e:
            # Mark failed
            with self._lock:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now().isoformat()
                task.error = str(e)
                task.progress = 0
            
            # Callback
            if self._on_task_fail:
                self._on_task_fail(task, e)
        
        finally:
            with self._lock:
                self._current_tasks.pop(worker_id, None)
    
    def _get_task_handler(self, task_type: TaskType) -> Callable:
        """Get handler function untuk task type"""
        handlers = {
            TaskType.PREPROCESSING: self._handle_preprocessing,
            TaskType.TRAINING: self._handle_training,
            TaskType.PREDICTION: self._handle_prediction,
            TaskType.EXPORT: self._handle_export,
        }
        return handlers.get(task_type, lambda t, p: None)
    
    def _handle_preprocessing(self, task: Task, progress: Callable) -> Any:
        """Handle preprocessing task"""
        from app.main_v3 import run_preprocessing
        # Note: This would need access to global state
        # Implementation depends on refactoring main_v3
        return {"status": "completed", "samples": 0}
    
    def _handle_training(self, task: Task, progress: Callable) -> Any:
        """Handle training task"""
        # Implementation depends on main_v3 refactoring
        # Would call run_training dengan config dari task.params
        return {"status": "completed", "models": []}
    
    def _handle_prediction(self, task: Task, progress: Callable) -> Any:
        """Handle prediction task"""
        # Fast prediction task
        return {"status": "completed"}
    
    def _handle_export(self, task: Task, progress: Callable) -> Any:
        """Handle export task (ONNX, dll)"""
        return {"status": "completed"}
    
    def _persist_task_completion(self, task: Task):
        """Persist task completion ke database"""
        if not self.persistence:
            return
        
        try:
            if task.type == TaskType.TRAINING:
                # Training session already handled di training code
                pass
            elif task.type == TaskType.PREPROCESSING:
                self.persistence.update_state(
                    preprocessed=True,
                    current_task=f"Preprocessing completed: {task.name}"
                )
        except Exception:
            pass
    
    def submit(
        self,
        task_type: TaskType,
        name: str,
        params: Dict[str, Any] = None,
        priority: int = 5
    ) -> str:
        """
        Submit task ke queue
        
        Args:
            task_type: Type of task
            name: Human-readable name
            params: Task parameters
            priority: Lower = higher priority (default 5)
        
        Returns:
            task_id: Unique task identifier
        """
        task_id = f"{task_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        task = Task(
            id=task_id,
            type=task_type,
            name=name,
            status=TaskStatus.PENDING,
            created_at=datetime.now().isoformat(),
            params=params or {}
        )
        
        with self._lock:
            self._tasks[task_id] = task
        
        # Add ke appropriate queue dengan priority
        self._queues[task_type].put((priority, task))
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_tasks(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        limit: int = 100
    ) -> List[Task]:
        """Get tasks dengan filtering"""
        with self._lock:
            tasks = list(self._tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        if task_type:
            tasks = [t for t in tasks if t.type == task_type]
        
        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        return tasks[:limit]
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel pending task
        Returns True jika berhasil cancel
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            # Can only cancel pending tasks
            if task.status != TaskStatus.PENDING:
                return False
            
            task.status = TaskStatus.CANCELLED
            return True
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            all_tasks = list(self._tasks.values())
            active = [t for t in all_tasks if t.status == TaskStatus.RUNNING]
            pending = [t for t in all_tasks if t.status == TaskStatus.PENDING]
            completed = [t for t in all_tasks if t.status == TaskStatus.COMPLETED]
            failed = [t for t in all_tasks if t.status == TaskStatus.FAILED]
        
        return {
            'total_tasks': len(all_tasks),
            'active': len(active),
            'pending': len(pending),
            'completed': len(completed),
            'failed': len(failed),
            'workers': self.max_workers,
            'current_tasks': self._current_tasks.copy()
        }
    
    def on_task_complete(self, callback: Callable[[Task], None]):
        """Set callback untuk task completion"""
        self._on_task_complete = callback
    
    def on_task_fail(self, callback: Callable[[Task, Exception], None]):
        """Set callback untuk task failure"""
        self._on_task_fail = callback
    
    def on_progress(self, callback: Callable[[str, int, str], None]):
        """Set callback untuk progress updates"""
        self._on_progress = callback
    
    def shutdown(self, wait: bool = True):
        """Shutdown task queue"""
        self._shutdown.set()
        
        if wait:
            for worker in self._workers:
                worker.join(timeout=5.0)


# Global task queue instance
_task_queue = None


def get_task_queue(max_workers: int = 2, persistence=None) -> TaskQueue:
    """Get atau create global TaskQueue instance"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue(max_workers=max_workers, persistence=persistence)
    return _task_queue


def shutdown_task_queue():
    """Shutdown global task queue"""
    global _task_queue
    if _task_queue:
        _task_queue.shutdown(wait=True)
        _task_queue = None
