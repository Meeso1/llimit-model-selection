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
        
    def inspect(self, max_depth: int = 0) -> None:
        self._inspect_internal(0, max_depth)
            
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
