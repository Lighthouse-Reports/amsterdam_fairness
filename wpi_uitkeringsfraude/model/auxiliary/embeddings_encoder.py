import logging
import os
import shutil
from datetime import datetime as dt
from io import BytesIO
from math import sqrt
from typing import Union

import h5py
import numpy as np
import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
from sklearn.base import TransformerMixin
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed

from wpi_onderzoekswaardigheid_aanvraag.project_paths import ARTIFACT_PATH

# Remove tensorflow GPU warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf_logger = logging.getLogger("tensorflow")
tf_logger.setLevel(logging.WARNING)


class EmbeddingsEncoder(TransformerMixin):
    """
    Encode into embeddings the categorical variables of the model.
    This is done applying an OrdinalEncoder to encode the categories into integers;
    and then a tensorflow model containing only one Embedding layer is fit and encodes the variable.
    """

    def __init__(self):
        set_seed(42)
        self._is_fit = False

    def fit(self, feature_col: pd.Series, target_col: pd.Series, n_epochs: int = 10):
        """Fit the encoder on the training set column.

        Parameters
        ----------
        feature_col
            series containing the training data
        target_col
            series containing the target
        n_epochs
            nr of epochs to train the embedding for (defaults to 10)

        Returns
        -------
        :
            object containing the training encoder

        """
        self._embed_feature(feature_col, target_col, scoring=False, n_epochs=n_epochs)
        self._is_fit = True
        return self

    def transform(self, feature_col: pd.Series, target_col: pd.Series = None) -> object:
        """Transform the column using the trained encoder.

        Parameters
        ----------
        feature_col
            series containing the data to encode
        target_col
            (not required) series containing the target

        Returns
        -------
        :
            dataframe containing the encoded data

        """
        return self._embed_feature(feature_col, target_col, scoring=True)

    def _embed_feature(
        self,
        feature_col: pd.Series,
        target_col: pd.Series,
        scoring: bool,
        n_epochs: int = 10,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Get embedded values for the dataframe column.

        Parameters
        ----------
        feature_col
            dataframe column containing the category to encode
        target_col
            series containing the target
        n_epochs
            nr of epochs to train the embedding for (defaults to 10)
        scoring
            If train or score

        Returns
        -------
        :
            dataframe containing the embeddings
        """

        data_ord_enc = self._prepare_input(feature_col, scoring)
        embeddings = self._calculate_embeddings(
            data_ord_enc, target_col, n_epochs, scoring
        )
        if scoring:
            result = self._create_df_columns(embeddings, data_ord_enc.name)
            return result
        return embeddings

    def _prepare_input(self, feature_col: pd.Series, scoring: bool) -> pd.Series:
        """Encode the series using an OrdinalEncoder.

        The missing and unknown values are classified as 0.
        If the input df is not a string is casted to string before doing the encoding.

        Parameters
        ----------
        feature_col
            series containing the data to encode

        Returns
        -------
        :
            series containing the encoded data
        """
        df_to_string = self._feature_to_clean_string(feature_col)
        if not scoring:
            self._train_ordinal_encoder(df_to_string)
        encoded = self.le_encoder_.transform(df_to_string)
        return encoded.fillna(0).iloc[:, 0]

    def _feature_to_clean_string(self, feature_col):
        """Cast the df column to string since we want to treat it as a categorical feature.
        Removes .0 from the end of the strings to make them look cleaner when they have been casted from float.

        Parameters
        ----------
        feature_col
            dataframe column containing the data to be casted

        Returns
        -------
        :
            casted and cleaned data
        """
        return feature_col.apply(str).str.rstrip(".0")

    def _train_ordinal_encoder(self, feature_strings):
        self.le_encoder_ = OrdinalEncoder(
            handle_unknown="return_nan", handle_missing="return_nan"
        )
        self.le_encoder_ = self.le_encoder_.fit(feature_strings)
        self.n_labels_ = len(self.le_encoder_.get_params()["mapping"][0]["mapping"])

    def _calculate_embeddings(
        self, data_ord_enc: pd.Series, target: pd.Series, n_epochs: int, scoring: bool
    ) -> np.ndarray:
        """
        Calculate embeddings from the ordinal encoded data.

        Parameters
        ----------
        data_ord_enc
            ordinal encoded series containing the data to embed
        target
            series containing the target

        Returns
        -------
        :
            array containing the embeddings
        """
        if not scoring:
            self._train_embeddings_model(data_ord_enc, target, n_epochs)
        output_array = self.embeddings_model_.predict(data_ord_enc)
        return output_array

    def _train_embeddings_model(self, data_ord_enc, target, n_epochs=10):
        self.n_target_classes_ = target.nunique()
        target_prepped = to_categorical(target, num_classes=self.n_target_classes_)
        target_prepped = target_prepped.reshape(
            target_prepped.shape[0], 1, target_prepped.shape[1]
        )

        checkpoint_filepath = (
            ARTIFACT_PATH / "tmp" / f"embedding_checkpoint_{dt.now()}.tf"
        )
        cb_store_best = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        )

        self.embeddings_model_ = Sequential()
        dimensions = round(sqrt(self.n_labels_))
        # +2 because when OrdinalEncoder finds a None, encodes as nan and skips one label
        self.embeddings_model_.add(
            Embedding(input_dim=self.n_labels_ + 2, output_dim=dimensions)
        )
        self.embeddings_model_.add(
            Dense(units=self.n_target_classes_, activation="softmax")
        )
        self.embeddings_model_.compile("rmsprop", "categorical_crossentropy")
        self.embeddings_model_.fit(
            data_ord_enc,
            target_prepped,
            epochs=n_epochs,
            validation_split=0.2,
            verbose=0,
            callbacks=[cb_store_best],
        )
        self.embeddings_model_ = load_model(checkpoint_filepath)
        shutil.rmtree(checkpoint_filepath)

    def _create_df_columns(
        self, embeddings: Union[list, np.ndarray], feature_name: str
    ) -> pd.DataFrame:
        """
        Create a dataframe from the array of embeddings.

        Parameters
        ----------
        embeddings
            array containing the embeddings

        Returns
        -------
        :
            dataframe containing the embeddings
        """
        embeddings_array = np.array(embeddings)
        embeddings_array = embeddings_array.reshape(
            [embeddings_array.shape[0], embeddings_array.shape[2]]
        )
        df = pd.DataFrame.from_records(embeddings_array)
        df = df.add_prefix(f"{feature_name}_")
        return df

    def __getstate__(self):
        """
        Replaces from the state that will be pickled the Keras model with a byte encoded version of it.

        Returns
        -------
        :
            state
        """
        state = self.__dict__.copy()
        if "embeddings_model_" in state:
            f = BytesIO()
            h5file = h5py.File(f, "w")
            state["embeddings_model_"].save(h5file)
            model_bytes = f.getvalue()
            state["embeddings_model_"] = model_bytes
        return state

    def __setstate__(self, state):
        """
        Replace from the pickled state the Keras byte encoded version with the model.
        """
        self.__dict__.update(state)
        if "embeddings_model_" in state:
            model_bytes = state["embeddings_model_"]
            f = BytesIO(model_bytes)
            h5file = h5py.File(f, "w")
            self.embeddings_model_ = load_model(h5file)

    def _raise_if_not_fitted(self):
        if not self._is_fit:
            raise ValueError("not fitted. Did you call fit_transform()?")
