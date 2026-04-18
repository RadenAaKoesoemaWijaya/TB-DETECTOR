"""
State Persistence Manager
SQLite-based persistence untuk pipeline state dan progress tracking
"""

import sqlite3
import json
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path


class PipelinePersistence:
    """
    SQLite-based persistence untuk TB Detector pipeline
    Menyimpan state, logs, dan training history
    """
    
    def __init__(self, db_path: str = "data/pipeline.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_db(self):
        """Initialize database tables"""
        with self._lock:
            conn = self._get_connection()
            
            # Pipeline state table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    dataset_uploaded BOOLEAN DEFAULT 0,
                    dataset_path TEXT,
                    preprocessed BOOLEAN DEFAULT 0,
                    preprocessed_samples INTEGER DEFAULT 0,
                    training_in_progress BOOLEAN DEFAULT 0,
                    current_task TEXT,
                    progress INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Logs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level TEXT DEFAULT 'INFO',
                    message TEXT,
                    task TEXT,
                    progress INTEGER
                )
            """)
            
            # Training sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    backbone_name TEXT,
                    config TEXT,  -- JSON
                    status TEXT DEFAULT 'running',  -- running, completed, failed
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    best_auroc REAL,
                    epochs_trained INTEGER,
                    model_path TEXT
                )
            """)
            
            # Training metrics per epoch
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    epoch INTEGER,
                    train_loss REAL,
                    train_acc REAL,
                    val_loss REAL,
                    val_auroc REAL,
                    val_f1 REAL,
                    learning_rate REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
                )
            """)
            
            # Model registry
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE,
                    backbone_name TEXT,
                    model_path TEXT,
                    metrics TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 0
                )
            """)
            
            # Cache statistics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backbone_name TEXT,
                    hits INTEGER DEFAULT 0,
                    misses INTEGER DEFAULT 0,
                    total_entries INTEGER DEFAULT 0,
                    total_size_mb REAL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert default state jika belum ada
            conn.execute("""
                INSERT OR IGNORE INTO pipeline_state (id) VALUES (1)
            """)
            
            conn.commit()
    
    def update_state(self, **kwargs):
        """Update pipeline state"""
        with self._lock:
            conn = self._get_connection()
            
            # Build update query
            allowed_fields = [
                'dataset_uploaded', 'dataset_path', 'preprocessed',
                'preprocessed_samples', 'training_in_progress',
                'current_task', 'progress'
            ]
            
            updates = []
            values = []
            for key, value in kwargs.items():
                if key in allowed_fields:
                    updates.append(f"{key} = ?")
                    values.append(value)
            
            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                query = f"UPDATE pipeline_state SET {', '.join(updates)} WHERE id = 1"
                conn.execute(query, values)
                conn.commit()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current pipeline state"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT * FROM pipeline_state WHERE id = 1")
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return {}
    
    def add_log(self, message: str, level: str = 'INFO', task: str = None, progress: int = None):
        """Add log entry"""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "INSERT INTO logs (level, message, task, progress) VALUES (?, ?, ?, ?)",
                (level, message, task, progress)
            )
            conn.commit()
            
            # Keep only last 1000 logs
            conn.execute("""
                DELETE FROM logs WHERE id NOT IN (
                    SELECT id FROM logs ORDER BY timestamp DESC LIMIT 1000
                )
            """)
            conn.commit()
    
    def get_logs(self, limit: int = 100, level: str = None) -> List[Dict[str, Any]]:
        """Get recent logs"""
        with self._lock:
            conn = self._get_connection()
            
            if level:
                cursor = conn.execute(
                    "SELECT * FROM logs WHERE level = ? ORDER BY timestamp DESC LIMIT ?",
                    (level, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def start_training_session(self, session_id: str, backbone_name: str, config: Dict) -> bool:
        """Start new training session"""
        with self._lock:
            try:
                conn = self._get_connection()
                conn.execute(
                    "INSERT INTO training_sessions (session_id, backbone_name, config) VALUES (?, ?, ?)",
                    (session_id, backbone_name, json.dumps(config))
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    def update_training_session(
        self,
        session_id: str,
        status: str = None,
        best_auroc: float = None,
        epochs_trained: int = None,
        model_path: str = None
    ):
        """Update training session"""
        with self._lock:
            conn = self._get_connection()
            
            updates = []
            values = []
            
            if status:
                updates.append("status = ?")
                values.append(status)
                if status in ['completed', 'failed']:
                    updates.append("end_time = CURRENT_TIMESTAMP")
            
            if best_auroc is not None:
                updates.append("best_auroc = ?")
                values.append(best_auroc)
            
            if epochs_trained is not None:
                updates.append("epochs_trained = ?")
                values.append(epochs_trained)
            
            if model_path:
                updates.append("model_path = ?")
                values.append(model_path)
            
            if updates:
                values.append(session_id)
                query = f"UPDATE training_sessions SET {', '.join(updates)} WHERE session_id = ?"
                conn.execute(query, values)
                conn.commit()
    
    def add_training_metrics(
        self,
        session_id: str,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_auroc: float,
        val_f1: float,
        learning_rate: float
    ):
        """Add training metrics for an epoch"""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """INSERT INTO training_metrics 
                    (session_id, epoch, train_loss, train_acc, val_loss, val_auroc, val_f1, learning_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, epoch, train_loss, train_acc, val_loss, val_auroc, val_f1, learning_rate)
            )
            conn.commit()
    
    def get_training_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get training metrics history"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM training_metrics WHERE session_id = ? ORDER BY epoch",
                (session_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_training_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent training sessions"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM training_sessions ORDER BY start_time DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            
            sessions = []
            for row in rows:
                session = dict(row)
                session['config'] = json.loads(session['config']) if session['config'] else {}
                sessions.append(session)
            
            return sessions
    
    def register_model(self, model_name: str, backbone_name: str, model_path: str, metrics: Dict):
        """Register trained model"""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """INSERT OR REPLACE INTO models 
                    (model_name, backbone_name, model_path, metrics) VALUES (?, ?, ?, ?)""",
                (model_name, backbone_name, model_path, json.dumps(metrics))
            )
            conn.commit()
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get all registered models"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT * FROM models ORDER BY created_at DESC")
            rows = cursor.fetchall()
            
            models = []
            for row in rows:
                model = dict(row)
                model['metrics'] = json.loads(model['metrics']) if model['metrics'] else {}
                models.append(model)
            
            return models
    
    def record_cache_stats(self, backbone_name: str, hits: int, misses: int, total_entries: int, total_size_mb: float):
        """Record cache statistics"""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """INSERT INTO cache_stats 
                    (backbone_name, hits, misses, total_entries, total_size_mb)
                    VALUES (?, ?, ?, ?, ?)""",
                (backbone_name, hits, misses, total_entries, total_size_mb)
            )
            conn.commit()
    
    def get_cache_stats(self, backbone_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get cache statistics history"""
        with self._lock:
            conn = self._get_connection()
            
            if backbone_name:
                cursor = conn.execute(
                    "SELECT * FROM cache_stats WHERE backbone_name = ? ORDER BY recorded_at DESC LIMIT ?",
                    (backbone_name, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM cache_stats ORDER BY recorded_at DESC LIMIT ?",
                    (limit,)
                )
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def clear_old_logs(self, days: int = 30):
        """Clear logs older than X days"""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "DELETE FROM logs WHERE timestamp < datetime('now', '-{} days')".format(days)
            )
            conn.commit()
    
    def reset_state(self):
        """Reset pipeline state ke default"""
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                UPDATE pipeline_state SET
                    dataset_uploaded = 0,
                    dataset_path = NULL,
                    preprocessed = 0,
                    preprocessed_samples = 0,
                    training_in_progress = 0,
                    current_task = NULL,
                    progress = 0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
            """)
            conn.commit()


# Global persistence instance
_persistence = None


def get_persistence(db_path: str = "data/pipeline.db") -> PipelinePersistence:
    """Get atau create global persistence instance"""
    global _persistence
    if _persistence is None:
        _persistence = PipelinePersistence(db_path)
    return _persistence
