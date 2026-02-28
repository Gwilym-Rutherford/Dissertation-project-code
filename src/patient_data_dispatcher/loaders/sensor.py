from .base import BaseLoader
from src.core.types import CSVData

class SensorLoader(BaseLoader):
    def __init__(self, config_path, metadata: CSVData):
        super().__init__(config_path)
        self.metadata = metadata
        
    def __call__(self, ids):
        pass

    