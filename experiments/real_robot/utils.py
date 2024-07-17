import numpy as np


class ImuFilter:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.data = []
        self.filtered_data = []

    def push_data(self, data):
        self.data.append(data)

    def get_filtered_data(self):
        data = self.data[-self.window_size :]
        return np.mean(data, axis=0)
