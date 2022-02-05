import logging
import math
import queue
import random
from queue import Queue
from threading import Timer


class OpponentSampler:
    def __init__(self, last_model, get_global_iteration, load_model, opponent_sampling_index):
        assert 0 < opponent_sampling_index <= 1

        self.last_model = last_model
        self.get_global_iteration = get_global_iteration
        self.load_model = load_model
        self.opponent_sampling_index = opponent_sampling_index
        self.prefetched_models = Queue()

        self.prefetch_models()

    def sample(self):
        num_of_models = self.get_global_iteration() - 1

        if num_of_models < 1:
            return self.last_model

        try:
            return self.prefetched_models.get_nowait()
        except queue.Empty:
            logging.warning("Loading model synchronously from disk because no prefetched model was available")
            return self.sample_from_disk()

    def prefetch_models(self):
        num_of_models = self.get_global_iteration() - 1

        while self.prefetched_models.qsize() < 50 and num_of_models > 0:
            self.prefetched_models.put(self.sample_from_disk())

        thread = Timer(function=self.prefetch_models, interval=1)
        thread.daemon = True
        thread.start()

    def sample_from_disk(self):
        num_of_models = self.get_global_iteration() - 1

        min_index = max(math.floor((1 - self.opponent_sampling_index) * num_of_models), 1)

        index = random.randint(min_index, num_of_models)
        return self.load_model(index)
