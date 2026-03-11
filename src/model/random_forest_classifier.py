from sklearn.ensemble import RandomForestClassifier
from numpy.typing import NDArray
from typing import Literal

class DMORandomForest:
    def __init__(
        self,
        n_trees: int = 100,
        critirion: Literal["gini", "entropy", "log_loss"] = "gini",
        max_depth: int | None = None,
        max_features: Literal["sqrt", "log2", None] = "sqrt",
        bootstrap: bool = True,
    ) -> None:
        self.random_forest = RandomForestClassifier(
            n_estimators=n_trees,
            criterion=critirion,
            max_depth=max_depth,
            max_features=max_features,
            bootstrap=bootstrap,
        )

    def train(self, input_data: NDArray, labels: NDArray) -> None:
        self.random_forest.fit(input_data, labels)

    def predict(self, input_data: NDArray) -> NDArray:
        return self.random_forest.predict(input_data)

    def score(self, input_data: NDArray, labels: NDArray) -> float:
        return self.random_forest.score(input_data, labels)

