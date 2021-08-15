class MovingAverageCalculator:
    def __init__(self, k):
        self.k = k
        self.items = []
        self.current_sum = 0

    def insert_value(self, value):
        self.items.append(value)
        self.current_sum += value

        if len(self.items) > self.k:
            self.current_sum -= self.items[0]
            del self.items[0]

    def get(self):
        return self.current_sum / len(self.items)
