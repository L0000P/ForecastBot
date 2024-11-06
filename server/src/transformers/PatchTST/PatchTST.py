import pandas as pd
import numpy as np
import os
import glob
import warnings
import torch
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from models.ForecastDFDataset import ForecastDFDataset
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
)

# Ignore Torch Warnings.
warnings.filterwarnings("ignore", module="torch")

class PatchTST:
    def __init__(self, log_path, model_path, context_length=512, forecast_horizon=96, patch_length=32, num_workers=8, batch_size=8):
        self.log_path = log_path
        self.model_path = model_path
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.patch_length = patch_length
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.scaler = RobustScaler()
        self.model = None
        self.trainer = None

        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

    def load_data(self, csv_files, timestamp_column="date", index_start=1):
        dataframes = [pd.read_csv(f, parse_dates=[timestamp_column]) for f in csv_files]
        dataset = pd.concat(dataframes, ignore_index=True)

        numeric_columns = dataset.select_dtypes(include=[np.number]).columns
        Q1 = dataset[numeric_columns].quantile(0.25)
        Q3 = dataset[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1

        dataset = dataset[~((dataset[numeric_columns] < (Q1 - 1.5 * IQR)) | (dataset[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

        dataset.ffill(inplace=True)
        dataset.bfill(inplace=True)

        dataset["day_of_week"] = dataset[timestamp_column].dt.dayofweek.astype(np.float32)
        dataset["day_of_month"] = dataset[timestamp_column].dt.day.astype(np.float32)
        dataset["month"] = dataset[timestamp_column].dt.month.astype(np.float32)
        dataset["hour"] = dataset[timestamp_column].dt.hour.astype(np.float32)

        scaled_values = self.scaler.fit_transform(dataset.iloc[:, index_start:])
        dataset.iloc[:, index_start:] = scaled_values.astype(np.float32)

        for col in dataset.columns[index_start:]:
            dataset[f'{col}_lag1'] = dataset[col].shift(1)
            dataset[f'{col}_rolling_mean'] = dataset[col].rolling(window=5).mean()
            dataset[f'{col}_rolling_std'] = dataset[col].rolling(window=5).std()

        dataset.ffill(inplace=True)
        dataset.bfill(inplace=True)

        num_train = int(len(dataset) * 0.7)
        num_valid = int(len(dataset) * 0.2)
        
        # Define datasets
        self.train_dataset = ForecastDFDataset(
            data_df=dataset.iloc[:num_train].reset_index(drop=True),
            timestamp_column=timestamp_column,
            target_columns=dataset.columns[index_start:],
            context_length=self.context_length,
            prediction_length=self.forecast_horizon
        )

        self.valid_dataset = ForecastDFDataset(
            data_df=dataset.iloc[num_train:num_train + num_valid].reset_index(drop=True),
            timestamp_column=timestamp_column,
            target_columns=dataset.columns[index_start:],
            context_length=self.context_length,
            prediction_length=self.forecast_horizon
        )

        self.test_dataset = ForecastDFDataset(
            data_df=dataset.iloc[num_train + num_valid:].reset_index(drop=True),
            timestamp_column=timestamp_column,
            target_columns=dataset.columns[index_start:],
            context_length=self.context_length,
            prediction_length=self.forecast_horizon
        )

    def configure_model(self):
        config = PatchTSTConfig(
            num_input_channels=len(self.train_dataset.target_columns),
            context_length=self.context_length,
            patch_length=self.patch_length,
            patch_stride=self.patch_length,
            prediction_length=self.forecast_horizon,
            random_mask_ratio=0.4,
            d_model=256,
            num_attention_heads=8,
            num_hidden_layers=6,
            ffn_dim=1024,
            dropout=0.2,
            head_dropout=0.2,
            pooling_type=None,
            channel_attention=True,
            scaling="standard",
            loss="mse",
            pre_norm=True,
            norm_type="layernorm",
        )
        self.model = PatchTSTForPrediction(config)

    def train(self, epochs=20, learning_rate=5e-6):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.model.to(device)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=0.01,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            dataloader_num_workers=self.num_workers,
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=3,
            logging_dir=self.log_path,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            label_names=["future_values"],
        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=20,
            early_stopping_threshold=0.0001,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            callbacks=[early_stopping_callback]
        )

        self.trainer.train()

        self.model.save_pretrained(self.model_path)
        print(f"Model saved to {self.model_path}")
        print("Files saved:", os.listdir(self.model_path))

    def load_model(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model path {self.model_path} does not exist.")
        self.model = PatchTSTForPrediction.from_pretrained(self.model_path)
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Model loaded from {self.model_path}")
        
    def predict(self, new_data, timestamp_column="date", output_csv="results/forecasted.csv"):
        Path("results").mkdir(parents=True, exist_ok=True)

        if self.model is None:
            self.load_model()

        if self.trainer is None:
            training_args = TrainingArguments(
                output_dir=self.model_path,
                per_device_eval_batch_size=self.batch_size,
                dataloader_num_workers=self.num_workers,
                logging_dir=self.log_path,
            )
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
            )

        new_data[timestamp_column] = pd.to_datetime(new_data[timestamp_column])
        new_data = new_data.sort_values(by=timestamp_column)

        new_data["day_of_week"] = new_data[timestamp_column].dt.dayofweek.astype(np.float32)
        new_data["day_of_month"] = new_data[timestamp_column].dt.day.astype(np.float32)
        new_data["month"] = new_data[timestamp_column].dt.month.astype(np.float32)
        new_data["hour"] = new_data[timestamp_column].dt.hour.astype(np.float32)

        for col in new_data.columns[1:]:  # Salta la colonna timestamp
            new_data[f'{col}_lag1'] = new_data[col].shift(1)
            new_data[f'{col}_rolling_mean'] = new_data[col].rolling(window=5).mean()
            new_data[f'{col}_rolling_std'] = new_data[col].rolling(window=5).std()

        new_data.fillna(method='ffill', inplace=True)
        new_data.fillna(method='bfill', inplace=True)

        expected_feature_names = self.scaler.get_feature_names_out()
        if not set(expected_feature_names).issubset(new_data.columns):
            raise ValueError("New data does not contain all the expected features.")

        new_data_filtered = new_data[expected_feature_names]
        new_data_scaled = self.scaler.transform(new_data_filtered).astype(np.float32)

        prepared_new_dataset = ForecastDFDataset(
            data_df=new_data.reset_index(drop=True),
            timestamp_column=timestamp_column,
            target_columns=new_data.columns[1:],  # Colonne target escludendo il timestamp
            context_length=self.context_length,
            prediction_length=self.forecast_horizon
        )

        predictions_tuple = self.trainer.predict(prepared_new_dataset)

        target_shape = (1, 96, 44)
        predictions_list = []
        for i, prediction in enumerate(predictions_tuple.predictions):
            if prediction.shape == target_shape:
                predictions_list.append(prediction)
            else:
                print(f"Warning: Prediction at index {i} has inconsistent shape {prediction.shape}, skipping.")

        if not predictions_list:
            raise ValueError("No valid predictions were generated with consistent shapes. Please check model configuration or input data.")

        predictions = np.concatenate(predictions_list, axis=0).flatten()
        prediction_length = min(len(predictions), len(new_data[timestamp_column]))

        results_df = pd.DataFrame({
            timestamp_column: new_data[timestamp_column].iloc[-prediction_length:].values,
            'Predictions': predictions[-prediction_length:]  # Regola le previsioni per corrispondere alla lunghezza del timestamp
        })

        results_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")

        return results_df