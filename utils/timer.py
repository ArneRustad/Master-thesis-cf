import time

class Timer:
    def __init__(self):
        self._start_time = {}
        self.elapsed_time = {}

    def start(self, name):
        """Start a new timer"""

        self._start_time[name] = time.perf_counter()

    def stop(self, name):
        """Stop the timer, and report the elapsed time"""
        curr_elapsed_time = time.perf_counter() - self._start_time[name]
        self.elapsed_time[name] = curr_elapsed_time
        self._start_time[name] = None
        print(f"Elapsed time for {name}: {curr_elapsed_time:0.3f} seconds")

    def save(self, path, save_dir = None):
        if not save_dir is None:
            path = os.path.join(save_dir, path)
            os.makedirs(save_dir, exist_ok = True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path, save_dir = None):
        if not save_dir is None:
            path = os.path.join(save_dir, path)
        with open(path, "rb") as f:
            return pickle.load(f)

def load_timer(path, save_dir = None):
    if not save_dir is None:
        path = os.path.join(save_dir, path)
    with open(path, "rb") as f:
        return pickle.load(f)