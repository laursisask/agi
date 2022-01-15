import json
import math


# http://www.johndcook.com/blog/standard_deviation/
class IncrementalStatsCalculator:
    def __init__(self):
        self.m = None
        self.s = None
        self.n = 0

    def append(self, x):
        self.n += 1

        if self.n == 1:
            self.m = x
            self.s = 0
        else:
            new_m = self.m + (x - self.m) / self.n
            self.s = self.s + (x - self.m) * (x - new_m)
            self.m = new_m

    def std(self):
        assert self.n >= 2
        return math.sqrt(self.s / (self.n - 1))

    def dump(self, file):
        json.dump({"m": self.m, "s": self.s, "n": self.n}, file)

    def load(self, file):
        saved_values = json.load(file)

        self.m = saved_values["m"]
        self.s = saved_values["s"]
        self.n = saved_values["n"]
