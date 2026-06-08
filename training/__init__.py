from training.config import TrainerConfig
from training.trainer import Trainer

# __all__ defines what gets imported when we write: from training import Trainer, ...
__all__ = ["Trainer", "TrainerConfig"]