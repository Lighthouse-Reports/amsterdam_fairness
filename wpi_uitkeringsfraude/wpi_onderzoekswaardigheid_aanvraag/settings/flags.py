from enum import IntFlag, auto, unique
from functools import reduce


@unique
class PipelineFlag(IntFlag):
    NONE = 0  # Required in case none of the other flags are active
    DEVELOPMENT_MODE = (
        auto()
    )  # Reduce dataset during training of the pipeline to speed it up
    GRIDSEARCH = auto()  # Perform gridsearch, else use hardcoded parameters
    SAVE_PARAMS = auto()  # Save parameters of the (best) model to parameters.yml
    BIAS = auto()  # Carry out bias analysis
    REWEIGH = auto()  # Reweigh biased features

    @classmethod
    def all(cls):
        return reduce(lambda m1, m2: m1 | m2, [m for m in cls.__members__.values()])
