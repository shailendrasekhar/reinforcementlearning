"""Learning rate and parameter schedulers."""


class LinearScheduler:
    """Linear interpolation from start to end over total_steps."""

    def __init__(self, start: float, end: float, total_steps: int):
        self.start = start
        self.end = end
        self.total_steps = max(total_steps, 1)
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        fraction = min(self.step_count / self.total_steps, 1.0)
        return self.start + fraction * (self.end - self.start)

    @property
    def value(self) -> float:
        fraction = min(self.step_count / self.total_steps, 1.0)
        return self.start + fraction * (self.end - self.start)


class ExponentialScheduler:
    """Exponential decay."""

    def __init__(self, start: float, end: float, decay_rate: float):
        self.current = start
        self.end = end
        self.decay_rate = decay_rate

    def step(self) -> float:
        self.current = max(self.end, self.current * self.decay_rate)
        return self.current

    @property
    def value(self) -> float:
        return self.current
