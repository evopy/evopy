import numpy as np


class Individual:
    def __init__(self, weight, strategy):
        self.weight = weight
        self.strategy = strategy
        self.fitness = None

    def reproduce(self):
        new_weight = self.weight + self.strategy * np.random.randn(len(self.weight))
        z = np.random.randn() / (2 * len(self.weight))
        new_strategy = max(self.strategy * np.exp(z), 0.01)
        return Individual(new_weight, new_strategy)
