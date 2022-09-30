import time

from torch import cuda


class CudaStopWatch:
    def __init__(self):
        self.start_event = cuda.Event(enable_timing=True, blocking=False)
        self.stop_event = cuda.Event(enable_timing=True, blocking=True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.stop_event.record()

    def elapsed_ms(self, iterations=1):
        return self.start_event.elapsed_time(self.stop_event) / iterations

    def elapsed_s(self):
        return self.elapsed_ms() / 1e3

    def elapsed_us(self):
        return self.elapsed_ms() * 1e3

    def elapsed_ns(self):
        return self.elapsed_ms() * 1e6


class StopWatch:
    def __init__(self):
        self.time = None

    def start(self):
        self.time = time.perf_counter_ns()

    def stop(self):
        return time.perf_counter_ns() - self.time

    def elapsed_ns(self):
        return time.perf_counter_ns() - self.time

    def elapsed_ms(self, itrs=1):
        return self.elapsed_ns() / (itrs * 1e6)

    def elapsed_s(self):
        return self.elapsed_ns() / 1e9

    def elapsed_us(self):
        return self.elapsed_ns() / 1e3


if __name__ == '__main__':
    pass
