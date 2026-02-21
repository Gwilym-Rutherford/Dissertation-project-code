from math import ceil
from datetime import datetime

import matplotlib.pyplot as plt
import os

class Grapher:
    def __init__(self, name: str, notes: str = "no notes", output_dir: str = "log output") -> None:
        self.name = name
        self.notes = notes
        self.output_dir = os.path.join(os.getcwd(), output_dir)
        self.graphs = {}

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)


    def log_values(self, values: list(tuple[str, float])) -> None:
        for key, value in values:
            if key not in self.graphs.keys():
                self.graphs[key] = []
            self.graphs[key].append(value)

    def make_graph(self):
        n_cols = 2
        n_rows = ceil(len(self.graphs.keys()) / n_cols)

        plt.figure(figsize=(15, n_rows * 4)) 
        for index, key in enumerate(self.graphs):
            
            plt.subplot(n_rows, n_cols, index + 1)

            y_values = self.graphs[key]
            x_values = list(range(0, len(y_values)))

            plt.plot(
                x_values,
                y_values,
            )

            plt.title(f"{self.name}-{key}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")

            plt.ylim(bottom=0)
            plt.grid()
        
        plt.figtext(0.5, 0.01, self.notes, wrap=True, horizontalalignment='center', fontsize=10)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        dir_path = os.path.join(self.output_dir, self.name)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        plt.savefig(os.path.join(dir_path, f"{timestamp}"))
        plt.show()
