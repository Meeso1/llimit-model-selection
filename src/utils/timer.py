import time
from typing import ClassVar, Literal, Self
from types import TracebackType


class Timer:
    default_verbosity: ClassVar[Literal["detailed", "start+end", "exit", "none"] | None] = "none"
    
    def __init__(self, name: str, verbosity: Literal["detailed", "start+end", "exit", "none"] = "exit", parent: Self | None = None) -> None:
        self.verbosity = self.default_verbosity if self.default_verbosity is not None else verbosity
        self.name = name
        self.parent = parent
        self.start_time: float | None = None
        self.elapsed_time: float | None = None
        self.parts: dict[str, Self] = {}

    def start(self) -> None:
        self.start_time = time.time()
        if self.verbosity == "start+end" or self.verbosity == "detailed":
            print(f"{self.name} started")

    def stop(self) -> float:
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.elapsed_time = time.time() - self.start_time
        if not self.verbosity == "none":
            print(f"{self.name} took {self._format_duration(self.elapsed_time)}")
            
        if self.verbosity == "detailed":
            for part_name, part_time in self.parts.items():
                print(f"\t{part_name} took {self._format_duration(part_time)}")
            
        if self.parent is not None:
            self.parent.parts[self.name] = self
            
        return self.elapsed_time
    
    def __enter__(self) -> "Timer":
        self.start()
        return self
    
    def __exit__(self, exc_type: type | None, exc_value: Exception | None, traceback: TracebackType | None) -> None:
        self.stop()
        
    def get_all_timings_recursive(self) -> dict[str, float]:
        """
        Get all timings from this timer and its children recursively.

        Returns:
            Flat dict mapping dotted paths (e.g. "train", "train.epoch_0", "train.epoch_0.perform_validation")
            to elapsed times in seconds. Only includes timers that have been stopped (have elapsed_time set).
        """
        result: dict[str, float] = {}
        self._collect_timings_recursive("", result)
        return result

    def inspect(self, max_depth: int = 0) -> None:
        self._inspect_internal(0, max_depth)

    def _collect_timings_recursive(self, prefix: str, result: dict[str, float]) -> None:
        if self.elapsed_time is not None:
            key = self.name if prefix == "" else f"{prefix}.{self.name}"
            result[key] = self.elapsed_time
        for _, part in self.parts.items():
            part_prefix = self.name if prefix == "" else f"{prefix}.{self.name}"
            part._collect_timings_recursive(part_prefix, result)

    def _inspect_internal(self, depth: int, max_depth: int) -> None:
        if depth > max_depth:
            return
        
        print(f"{'  ' * depth}{self.name} took {self._format_duration(self.elapsed_time)}")
        for _, part in self.parts.items():
            part._inspect_internal(depth + 1, max_depth)
            
    def _format_duration(self, duration: float) -> str:
        if duration < 1e-4:
            return f"{duration:.2e} s"
        elif duration < 1e-2:
            return f"{duration * 1000:.2f} ms"
        elif duration < 1:
            return f"{duration * 1000:.0f} ms"
        elif duration < 10:
            return f"{duration:.3f} s"
        elif duration < 100:
            return f"{duration:.2f} s"
        else:
            return f"{duration:.0f} s"
