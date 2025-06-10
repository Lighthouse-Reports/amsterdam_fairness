from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fraude_preventie.optimize_dtypes import optimize_dtypes
from fraude_preventie.supertimer import timer

logger = logging.getLogger(__name__)


class Component:
    """A component is a generalization and subset of a pipeline transformer known from
    e.g. sklearn.

    A component has a :meth:`.fit` and a :meth:`.transform` function. That's all
    requirements. Both have an arbitrary amount of input arguments, and transform also
    has an arbitrary amount of return values.

    Usually both get one or more datasets as input. :meth:`.fit` learns any parameters
    necessary to transform unseen data.

    :meth:`.transform` transform new data based on the parameters learned in fit().

    The base class :class:`.Component` does not have default implementations of either
    method and throws a :class:`.NotImplementedError` when trying to call either.
    Subclasses must provide a valid implementation.

    Subclassing
    -----------
    Subclasses must implement the :meth:`_fit` and :meth:`._transform` functions.
    Please note the underscore -- the public :meth:`.fit` and :meth:`.transform`
    functions should not be overriden as they provide some boilerplate code such as
    logging, updating the :attr:`.is_fitted` indicator etc.

    Subclasses may require more specific arguments that must be provided to fit and
    transform.

    See Also
    --------
    * :class:`.NoFittingRequiredMixing`
    """

    def __init__(self):
        self.is_fitted = False
        self.fit_artifacts = {}

    def fit(self, *args, **kwargs) -> Component:
        """Fit the Component to the data passed via the arguments.

        Parameters
        ----------
        args
            positional arguments. Forwarded to subclasses, which can specify the
            requirements.
        kwargs
            keyword arguments. Forwarded to subclasses, which can specify the
            requirements.
        """

        name = self.__class__.__name__
        with timer(f"Fit {name}", loglevel=logging.INFO):
            self.fit_artifacts = self._fit(*args, **kwargs)
        self.is_fitted = True
        return self

    def transform(
        self, scoring: bool, do_dtype_optimization: bool = True, *args, **kwargs
    ):
        """Transform the input data using parameters learned during fitting.

        Raises an exception if the component has not been fit.

        Parameters
        ----------
        scoring
            indicate whether the function is called during scoring or not. Some
            components may choose to do slightly different preprocessing or handle a
            different column set such as the label during training than during scoring.
        do_dtype_optimization
            whether to downcast dtypes of the transformed result to minimize memory usage
        args
            positional arguments. Forwarded to subclasses, which can specify the
            requirements.
        kwargs
            keyword arguments. Forwarded to subclasses, which can specify the
            requirements.

        Returns
        -------
        :
            one or more values. It's up to the subclass to return whatever they want.
            Often it will be a single, transformed dataframe.
        """
        self._raise_if_not_fitted()

        with timer(f"Apply {self.__class__.__name__}", loglevel=logging.INFO):
            res = self._transform(scoring, *args, **kwargs)
            if isinstance(res, pd.DataFrame):
                if do_dtype_optimization:
                    res = optimize_dtypes(res)
                logger.debug(f"Shape is {res.shape}")
            return res

    def fit_transform(self, *args, **kwargs):
        """Call fit() and then transform() on the input arguments. The scoring
        argument of transform() is automatically set to ``False``."""
        self.fit(*args, **kwargs)
        return self.transform(scoring=False, *args, **kwargs)

    def _fit(self, *args, **kwargs) -> Any:
        """Fits this component. Must return all learned parameters, transformers etc.

        The format and structure of the artifacts can be freely determined by subclasses.

        Must be implemented by subclasses."""
        raise NotImplementedError

    def _transform(self, scoring: bool, *args, **kwargs) -> Any:
        """Must be implemented by subclasses"""
        raise NotImplementedError

    def _raise_if_not_fitted(self):
        """Raise an exception if this Component is not fitted."""
        if not self.is_fitted:
            raise ValueError("not fitted")


class NoFittingRequiredMixin:
    """Mixin for :class:`.Component`s that do not need fitting.

    :meth:`.fit` does not need to be called by subclasses using this mixin.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_fitted = True

    def _fit(self, *args, **kwargs):
        """Do nothing."""
        return None
