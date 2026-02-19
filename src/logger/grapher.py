import matplotlib.pyplot as plt
from math import ceil


class Grapher:
    def __init__(self, name: str, output_dir: str = "/log output") -> None:
        self.name = name
        self.output_dir = output_dir
        self.graphs = {}

    def log_values(self, values: list(tuple[str, float])) -> None:
        for key, value in values:
            if key not in self.graphs.keys():
                self.graphs[key] = []
            self.graphs[key].append(value)

    def make_graph(self):
        n_cols = 2
        n_rows = ceil(len(self.graphs.keys()) / n_cols)

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

        plt.show()
