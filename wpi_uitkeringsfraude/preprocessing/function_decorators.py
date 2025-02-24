import logging
from functools import wraps

import pandas as pd

logger = logging.getLogger(__name__)


def log_filtering_step(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            df = kwargs["df"]
        except KeyError:
            if isinstance(args[1], pd.DataFrame):
                df = args[1]
            else:
                logger.info(
                    f"Decorated function {f.__name__} is missing required argument `df`, "
                    f"decorator has no effect"
                )
                return f(*args, *kwargs)

        n_before = len(df)
        result = f(*args, **kwargs)
        logger.debug(
            f"{f.__name__}:Dropped {n_before - len(result)} rows",
        )
        return result

    return wrapper
