"""Training logger that saves logs to disk using the jar system."""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import time

from src.utils.jars import Jars
from src.utils.timer import Timer


@dataclass
class LogEntry:
    """A single log entry (typically per epoch)."""
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingLog:
    """Container for all training logs for a run."""
    run_name: str
    config: dict[str, Any]
    epoch_logs: list[LogEntry] = field(default_factory=list)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    timings: dict[str, float] | None = None


@dataclass
class RunInfo:
    """Information about a training run."""
    base_name: str
    timestamps: list[datetime]
    latest_timestamp: datetime
    latest_full_name: str
    
    
class TrainingLogger:
    """
    Logger for training runs that replaces wandb integration.
    
    Provides a similar interface to wandb but stores logs locally using the jar system.
    Each run gets a unique name with timestamp, and only the latest version is kept.
    """
    
    def __init__(self, run_name: str) -> None:
        """
        Initialize a training logger.
        
        Args:
            run_name: Name for this training run (timestamp will be appended)
        """
        self._run_name = run_name
        self._current_run: TrainingLog | None = None
    
    def init(self, config: dict[str, Any]) -> None:
        """
        Initialize a new training run.
        
        Args:
            config: Configuration dictionary for this run
        """
        # Append timestamp to run name to make it unique
        timestamp = int(time.time())
        unique_run_name = f"{self._run_name}-{timestamp}"
        
        self._current_run = TrainingLog(
            run_name=unique_run_name,
            config=config,
        )
    
    @property
    def run_name(self) -> str:
        """Get the run name (without timestamp)."""
        return self._run_name
    
    @property
    def full_run_name(self) -> str | None:
        """Get the full run name (with timestamp)."""
        if self._current_run is None:
            return None
        return self._current_run.run_name
    
    def log(self, data: dict[str, Any], log_timings_from: Timer | None = None) -> None:
        """
        Log data for the current epoch/step.
        
        Args:
            data: Dictionary of metrics to log
            log_timings_from: Optional timer to extract timings from (overwrites timings field)
        """
        if self._current_run is None:
            raise RuntimeError("No active training run. Call init() first.")

        if log_timings_from is not None:
            self._current_run.timings = log_timings_from.get_all_timings_recursive()

        entry = LogEntry(data=data)
        self._current_run.epoch_logs.append(entry)
        
        self._save()
    
    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Log final metrics for the run (e.g., test accuracy, final model scores).
        
        Args:
            metrics: Dictionary of final metrics
        """
        if self._current_run is None:
            raise RuntimeError("No active training run. Call init() first.")

        if len(self._current_run.final_metrics) > 0:
            warnings.warn("Final metrics have been logged. They will be updated.")
        
        self._current_run.final_metrics.update(metrics)
        self._save()
    
    def finish(self, log_timings_from: Timer | None = None) -> None:
        """
        Finish the current training run.

        Args:
            log_timings_from: Optional timer to extract timings from (overwrites timings field)
        """
        if self._current_run is None:
            return

        if log_timings_from is not None:
            self._current_run.timings = log_timings_from.get_all_timings_recursive()

        self._current_run.end_time = datetime.now()
        self._save()
        self._current_run = None
    
    def _save(self) -> None:
        """Save the current run to disk."""
        if self._current_run is None:
            return
        
        Jars.training_logs.replace(self._current_run.run_name, self._current_run)


def load_training_log(base_name: str, timestamp: int | str | None = None) -> TrainingLog:
    """
    Load a training log from disk.
    
    Args:
        base_name: Base name of the run (without timestamp)
        timestamp: Optional timestamp. If None, loads the latest run with this base name.
                   Can be int (unix timestamp) or str (timestamp as string).
        
    Returns:
        The loaded training log
    """
    if timestamp is not None:
        # Load specific run by full name
        timestamp_str = str(timestamp)
        full_name = f"{base_name}-{timestamp_str}"
        return Jars.training_logs.get(full_name)
    
    # Load latest run with this base name
    all_runs = list_training_runs()
    matching_runs = [
        run_info for run_info in all_runs
        if run_info.base_name == base_name
    ]
    
    if not matching_runs:
        raise KeyError(f"No training log found with base name '{base_name}'")

    if len(matching_runs) > 1:
        raise Exception("Multiple training logs have the same base name - this shouldn't happen")
    
    return Jars.training_logs.get(matching_runs[0].latest_full_name)


def list_training_runs() -> list[RunInfo]:
    """
    List all available training runs grouped by base name.
    
    Returns:
        List of RunInfo objects, sorted by latest timestamp (newest first)
    """
    all_names = Jars.training_logs.object_names()
    
    runs_by_base_name: dict[str, list[tuple[str, datetime]]] = {}
    for full_name in all_names:
        # Split on last occurrence of '-' to separate base_name and timestamp
        last_dash_idx = full_name.rfind('-')
        if last_dash_idx == -1:
            warnings.warn(f"No timestamp separator found for run {full_name}")
            continue
        
        base_name = full_name[:last_dash_idx]
        timestamp_str = full_name[last_dash_idx + 1:]
        
        try:
            # Convert unix timestamp to datetime
            timestamp_int = int(timestamp_str)
            timestamp = datetime.fromtimestamp(timestamp_int)
            
            if base_name not in runs_by_base_name:
                runs_by_base_name[base_name] = []
            runs_by_base_name[base_name].append((full_name, timestamp))
        except ValueError:
            warnings.warn(f"Invalid timestamp format for run {full_name}: {timestamp_str}")
            continue
    
    run_infos = []
    for base_name, runs in runs_by_base_name.items():
        runs.sort(key=lambda x: x[1], reverse=True)
        timestamps = [ts for _, ts in runs]
        latest_full_name, latest_timestamp = runs[0]
        
        run_infos.append(RunInfo(
            base_name=base_name,
            timestamps=timestamps,
            latest_timestamp=latest_timestamp,
            latest_full_name=latest_full_name,
        ))
    
    run_infos.sort(key=lambda r: r.latest_timestamp, reverse=True)
    return run_infos
