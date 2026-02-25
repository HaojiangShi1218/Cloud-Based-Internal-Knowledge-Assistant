import time
from contextlib import contextmanager

@contextmanager
def span(name: str, spans: dict):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        spans[name] = round((time.perf_counter() - t0) * 1000, 1)
