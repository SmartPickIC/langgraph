import queue

# 글로벌 변수로 큐 선언
_global_queue = None

def get_queue():
    global _global_queue
    if _global_queue is None:
        _global_queue = queue.Queue()
    return _global_queue

def add_log(message):
    get_queue().put(message)