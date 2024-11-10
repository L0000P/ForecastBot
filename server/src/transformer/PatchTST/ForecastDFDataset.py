import pandas as pd
import copy
from typing import List
from .BaseDFDataset import BaseDFDataset, np_to_torch

class ForecastDFDataset(BaseDFDataset):
    def __init__(
        self,
        data_df: pd.DataFrame,          # DataFrame containing the data
        timestamp_column: str,          # Name of the column containing the timestamps
        context_length: int = 1,        # Number of time steps to consider as context
        prediction_length: int = 1,     # Number of time steps to predict
        target_columns: List[str] = []   # List of column names for the target features            
    ):
        self.target_columns = list(target_columns)  # List of target feature column names
        x_cols = self.target_columns                 # List of input feature column names
        y_cols = copy.copy(x_cols)                  # List of target feature column names

        super().__init__(                             # Initialize the base class
            data_df=data_df,
            timestamp_column=timestamp_column,
            x_cols=x_cols,
            y_cols=y_cols,
            context_length=context_length,
            prediction_length=prediction_length
        )

        # Validate context_length and prediction_length
        if context_length <= 0 or prediction_length <= 0:
            raise ValueError("context_length and prediction_length must be positive integers.")
        
        # Ensure the DataFrame contains the necessary columns
        missing_columns = set(x_cols + y_cols) - set(data_df.columns)
        if missing_columns:
            raise ValueError(f"The following columns are missing from the DataFrame: {missing_columns}")

    def __getitem__(self, time_id: int) -> dict:
        # Reset the DataFrame index to ensure continuous integer index
        if not self.data_df.index.is_unique:
            self.data_df = self.data_df.reset_index(drop=True)
        
        s_pos = time_id
        e_pos = time_id + self.context_length

        # Ensure indices are within bounds
        if e_pos > len(self.data_df) or e_pos + self.prediction_length > len(self.data_df):
            raise IndexError("Index out of bounds for DataFrame.")

        # Get past values and future values
        seq_x = self.data_df.loc[s_pos:e_pos - 1, self.x_cols].values
        seq_y = self.data_df.loc[e_pos:e_pos + self.prediction_length - 1, self.y_cols].values

        return {
            "past_values": np_to_torch(seq_x),
            "future_values": np_to_torch(seq_y)
        }

    def __len__(self) -> int:  # Get the length of the dataset
        return len(self.data_df) - self.context_length - self.prediction_length + 1  # Return the length of the dataset