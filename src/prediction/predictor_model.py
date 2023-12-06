import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict
from darts.models.forecasting.transformer_model import TransformerModel
from darts import TimeSeries
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from torch import cuda
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"
MODEL_FILE_NAME = "model.joblib"


class Forecaster:
    """A wrapper class for the Transformer Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "Transformer Forecaster"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        input_chunk_length: int,
        output_chunk_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: Optional[str] = None,
        optimizer_kwargs: Optional[Dict] = None,
        random_state: Optional[int] = 0,
        **kwargs,
    ):
        """Construct a new Transformer Forecaster

        Args:
            input_chunk_length (int):
                Number of time steps in the past to take as a model input (per chunk). Applies to the target series,
                and past and/or future covariates (if the model supports it).

            output_chunk_length (int):
                Number of time steps predicted at once (per chunk) by the internal model.
                Also, the number of future values from future covariates to use as a model input (if the model supports future covariates).
                It is not the same as forecast horizon n used in predict(), which is the desired number of
                prediction points generated using either a one-shot- or auto-regressive forecast.
                Setting n <= output_chunk_length prevents auto-regression.
                This is useful when the covariates don't extend far enough into the future,
                or to prohibit the model from using future values of past and / or future covariates for prediction
                (depending on the model's covariate support).

            d_model (int):
                The number of expected features in the transformer encoder/decoder inputs (default=64).

            nhead (int):
                The number of heads in the multi-head attention mechanism (default=4).

            num_encoder_layers (int):
                The number of encoder layers in the encoder (default=3).

            num_decoder_layers (int):
                The number of decoder layers in the decoder (default=3).

            dim_feedforward (int):
                The dimension of the feedforward network model (default=512).

            dropout (float):
                Fraction of neurons affected by Dropout (default=0.1).

            activation (str):
                The activation function of encoder/decoder intermediate layer, (default='relu').
                can be one of the glu variant's FeedForward Network (FFN).
                A feedforward network is a fully-connected layer with an activation.
                The glu variant's FeedForward Network are a series of FFNs designed to work better with Transformer based models.
                [“GLU”, “Bilinear”, “ReGLU”, “GEGLU”, “SwiGLU”, “ReLU”, “GELU”] or one the pytorch internal activations [“relu”, “gelu”]

            norm_type (Optional[str]):
                The type of LayerNorm variant to use.
                Default: None. Available options are [“LayerNorm”, “RMSNorm”, “LayerNormNoBias”].

            optimizer_kwargs (Optional[dict]):
                Optionally, some keyword arguments for the PyTorch optimizer (e.g., {'lr': 1e-3} for specifying a learning rate).
                Otherwise the default values of the selected optimizer_cls will be used. Default: None.

            random_state (int): Sets the underlying random seed at model initialization time.


            **kwargs: Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and Darts' TorchForecastingModel.
        """
        self.data_schema = data_schema
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.norm_type = norm_type
        self.optimizer_kwargs = optimizer_kwargs
        self.random_state = random_state
        self._is_trained = False
        self.kwargs = kwargs

        if not data_schema.past_covariates:
            self.lags_past_covariates = None

        if not data_schema.future_covariates:
            self.lags_future_covariates = None

        self.history_length = None
        if kwargs.get("history_length"):
            self.history_length = kwargs["history_length"]
            kwargs.pop("history_length")

        pl_trainer_kwargs = None
        if cuda.is_available():
            pl_trainer_kwargs = {
                "accelerator": "gpu",
            }
            print("GPU training is available.")
        else:
            print("GPU training not available.")

        self.model = TransformerModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            norm_type=self.norm_type,
            optimizer_kwargs=self.optimizer_kwargs,
            pl_trainer_kwargs=pl_trainer_kwargs,
            **kwargs,
        )

    def _prepare_data(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
        history_length: int = None,
        test_dataframe: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Puts the data into the expected shape by the forecaster.
        Drops the time column and puts all the target series as columns in the dataframe.

        Args:
            history (pd.DataFrame): The provided training data.
            data_schema (ForecastingSchema): The schema of the training data.

        Returns:
            pd.DataFrame: The processed data.
        """
        targets = []
        past = []
        future = []

        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        self.all_ids = all_ids
        scalers = {}
        for index, s in enumerate(all_series):
            if history_length:
                s = s.iloc[-self.history_length :]
            s.reset_index(inplace=True)

            past_scaler = MinMaxScaler()
            scaler = MinMaxScaler()
            s[data_schema.target] = scaler.fit_transform(
                s[data_schema.target].values.reshape(-1, 1)
            )

            scalers[index] = scaler

            target = TimeSeries.from_dataframe(s, value_cols=data_schema.target)
            targets.append(target)

            if data_schema.past_covariates:
                original_values = (
                    s[data_schema.past_covariates].values.reshape(-1, 1)
                    if len(data_schema.past_covariates) == 1
                    else s[data_schema.past_covariates].values
                )
                s[data_schema.past_covariates] = past_scaler.fit_transform(
                    original_values
                )
                past_covariates = TimeSeries.from_dataframe(
                    s[data_schema.past_covariates]
                )
                past.append(past_covariates)

        if data_schema.future_covariates:
            test_groups_by_ids = test_dataframe.groupby(data_schema.id_col)
            test_all_series = [
                test_groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
                for id_ in all_ids
            ]

            for train_series, test_series in zip(all_series, test_all_series):
                if history_length:
                    train_series = train_series.iloc[-self.history_length :]
                    test_series = test_series.iloc[-self.history_length :]

                train_future_covariates = train_series[data_schema.future_covariates]
                test_future_covariates = test_series[data_schema.future_covariates]
                future_covariates = pd.concat(
                    [train_future_covariates, test_future_covariates], axis=0
                )
                future_covariates.reset_index(inplace=True)
                future_scaler = MinMaxScaler()
                original_values = (
                    future_covariates[data_schema.future_covariates].values.reshape(
                        -1, 1
                    )
                    if len(data_schema.future_covariates) == 1
                    else future_covariates[data_schema.future_covariates].values
                )
                future_covariates[
                    data_schema.future_covariates
                ] = future_scaler.fit_transform(original_values)
                future_covariates = TimeSeries.from_dataframe(future_covariates)
                future.append(future_covariates)

        self.scalers = scalers
        if not past:
            past = None
        if not future:
            future = None
        return targets, past, future

    def fit(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
        history_length: int = None,
        test_dataframe: pd.DataFrame = None,
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate Transformer model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
            data_schema (ForecastingSchema): The schema of the training data.
            history_length (int): The length of the series used for training.
            test_dataframe (pd.DataFrame): The testing data (needed only if the data contains future covariates).
        """
        np.random.seed(self.random_state)
        targets, past_covariates, future_covariates = self._prepare_data(
            history=history,
            history_length=history_length,
            data_schema=data_schema,
            test_dataframe=test_dataframe,
        )
        self.model.fit(
            targets,
            past_covariates=past_covariates,
        )
        self._is_trained = True
        self.data_schema = data_schema
        self.targets_series = targets
        self.past_covariates = past_covariates
        self.future_covariates = future_covariates

    def predict(
        self, test_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            pd.DataFrame: The predictions dataframe.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        predictions = self.model.predict(
            n=self.data_schema.forecast_length,
            series=self.targets_series,
            past_covariates=self.past_covariates,
        )
        prediction_values = []
        for index, prediction in enumerate(predictions):
            prediction = prediction.pd_dataframe()
            values = prediction.values
            values = self.scalers[index].inverse_transform(values)
            prediction_values += list(values)

        test_data[prediction_col_name] = np.array(prediction_values)
        return test_data

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        self.model.save(os.path.join(model_dir_path, MODEL_FILE_NAME))
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        forecaster = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        model = TransformerModel.load(os.path.join(model_dir_path, MODEL_FILE_NAME))
        forecaster.model = model
        return forecaster

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
    testing_dataframe: pd.DataFrame = None,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.
        test_dataframe (pd.DataFrame): The testing data (needed only if the data contains future covariates).

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(
        history=history,
        data_schema=data_schema,
        history_length=model.history_length,
        test_dataframe=testing_dataframe,
    )
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
