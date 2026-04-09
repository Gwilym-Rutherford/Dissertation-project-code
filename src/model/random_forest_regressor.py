from sklearn.ensemble import RandomForestRegressor
from numpy.typing import NDArray
from typing import Literal

class DMORandomForestRegressor:
    def __init__(
        self,
        n_trees: int = 100,
        critirion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = "squared_error",
        max_depth: int | None = None,
        max_features: Literal["sqrt", "log2", int, float, None] = 1.0,
        bootstrap: bool = True,
    ) -> None:
        self.random_forest = RandomForestRegressor(
            n_estimators=n_trees,
            criterion=critirion,
            max_depth=max_depth,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=-1
        )

    def train(self, input_data: NDArray, labels: NDArray) -> None:
        self.random_forest.fit(input_data, labels)

    def predict(self, input_data: NDArray) -> NDArray:
        return self.random_forest.predict(input_data)

    def score(self, input_data: NDArray, labels: NDArray) -> float:
        return self.random_forest.score(input_data, labels)

