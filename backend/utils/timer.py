import time
from typing import Dict, Any

class StageTimer:
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, stage_name: str):
        self._start_times[stage_name] = time.perf_counter()

    def stop(self, stage_name: str) -> float:
        if stage_name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[stage_name]
            self.timings[stage_name] = round(elapsed, 3)
            return self.timings[stage_name]
        return 0.0

    def get_summary(self) -> Dict[str, float]:
        return self.timings