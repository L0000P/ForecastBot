import pandas as pd
import numpy as np
import torch
import math
from datetime import datetime 

class MultivariateTSDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 timestamp_column: str,
                 training_columns: list[str],
                 target_columns: list[str],
                 context_length: int = 1,
                 prediction_length: int = 1,
                 iszeropadding: bool = True):
        super().__init__()
        self.timestamp_column = timestamp_column # Time Stamp Column
        self.training_columns = self.check_arg_if_list(training_columns) # Training Columns
        self.target_columns = self.check_arg_if_list(target_columns) # Target Columns
        self.context_length = context_length # Context Length
        self.prediction_length = prediction_length # Prediction Length

        data_df[timestamp_column] = pd.to_datetime(data_df[timestamp_column]) # Convert to DateTime
        self.data_df = data_df.sort_values(timestamp_column, ignore_index=True) # Sort Values

        if iszeropadding: # Zero Padding
            self.zero_padding() # Zero Padding
    
    @staticmethod 
    def date_to_vector(date): # Date to Vector
        dt = datetime.strptime(date, '%Y-%m-%d %H:%M:%S') # Date Time
        year = dt.year # Year
        month = dt.month # Month
        day = dt.day # Day
        day_of_week = dt.weekday()  # Day of Week
        
        year_vector = [
            math.cos(2 * math.pi * year / 365),
            math.sin(2 * math.pi * year / 365)
        ]
        
        month_vector = [
            math.cos(2 * math.pi * month / 12),
            math.sin(2 * math.pi * month / 12)
        ]
        
        day_vector = [
            math.cos(2 * math.pi * day / 31),
            math.sin(2 * math.pi * day / 31)
        ]
        
        day_of_week_vector = [
            math.cos(2 * math.pi * day_of_week / 7),
            math.sin(2 * math.pi * day_of_week / 7)
        ]

        date_vector = np.array(year_vector + month_vector + day_vector + day_of_week_vector)
        return date_vector

    @staticmethod
    # Check if Argument is List
    def check_arg_if_list(value):
        if isinstance(value, list):
            return value
        else:
            return [value]

    @staticmethod
    # Convert Numpy Array to PyTorch Tensor
    def np_to_torch(data: np.array, float_type=np.float32):
        data = np.nan_to_num(data) # Convert NaN to Num
        if data.dtype in ["float32", "float64"]: # If Data Type is Float32 or Float64
            return torch.from_numpy(data.astype(float_type)) # Convert Data to PyTorch Tensor
        elif data.dtype in ["int32", "int64"]: 
            return torch.from_numpy(data) # Convert Data to PyTorch Tensor
        else:
            raise TypeError(f"Unsupported data type: {data.dtype}")

    # Zero Padding
    def zero_padding(self):
        df_size = len(self.data_df) # Length of Data Frame
        if df_size >= self.context_length: # If Data Frame Size is Greater than Context Length
            return

        fill_length = self.context_length - df_size # Fill Length
        zeros_nump = np.zeros([fill_length, self.data_df.shape[1]]) # Zeros Numpy
        df_padding = pd.DataFrame(zeros_nump, columns=self.data_df.columns) # Data Frame Padding

        for column in self.data_df.columns: # For Column in Data Frame Columns
            if column == self.timestamp_column: # If Column is Time Stamp Column
                continue
            df_padding[column] = df_padding.astype(self.data_df[column].dtype, copy=False) # Copy Data Type

        date_1 = self.data_df.iloc[0][self.timestamp_column] # Date 1
        date_2 = self.data_df.iloc[1][self.timestamp_column] # Date 2
        period = date_2 - date_1 # Period

        new_timestamp = [ # New Time Stamp
            date_1 + offset * period 
            for offset in range(-fill_length, 0)
        ]
        df_padding[self.timestamp_column] = new_timestamp # Time Stamp Column

        self.data_df = pd.concat([df_padding, self.data_df]) # Concatenate Data Frame

    # Length
    def __len__(self):
        return len(self.data_df) - self.context_length - self.prediction_length + 1
    
    # Get Item
    def __getitem__(self, time_id):
        start_train_pos = time_id # Start Train Position
        end_train_pos   = time_id       + self.context_length # End Train Position
        end_predict_pos = end_train_pos + self.prediction_length # End Predict Position
       
        training_window = self.data_df.loc[ # Training Window
            start_train_pos : end_train_pos-1,   
            self.training_columns
        ].values
            
        prediction_window = self.data_df.loc[ # Prediction Window
            end_train_pos : end_predict_pos-1,
            self.target_columns
        ].values

        training_window_torch = self.np_to_torch(training_window) # Training Window Torch
        prediction_window_torch = self.np_to_torch(prediction_window) # Prediction Window Torch

        feature_time_mtx_past   = torch.randn((self.context_length    , training_window_torch.shape[1])) # Feature Time Matrix Past
        feature_time_mtx_future = torch.randn((self.prediction_length , prediction_window_torch.shape[1])) # Feature Time Matrix Future
        
        ft_time_past = self.date_to_vector(str(self.data_df.loc[time_id][self.timestamp_column]))

        ft_time_past   = torch.FloatTensor(np.array([MultivariateTSDataset.date_to_vector(str(self.data_df.loc[time_id][self.timestamp_column])) for time_id_item in range(self.context_length)]))
        ft_time_future = torch.FloatTensor(np.array([MultivariateTSDataset.date_to_vector(str(self.data_df.loc[time_id][self.timestamp_column])) for time_id_item in range(self.prediction_length)]))
        
        past_observed_mask = torch.ones_like(training_window_torch)

        ret = {
            "past_values"  : training_window_torch,
            "future_values": prediction_window_torch,
            
            "past_observed_mask"   : past_observed_mask,
            "past_time_features"   : feature_time_mtx_past,
            "future_time_features" : feature_time_mtx_future,

            "past_time_features"   : ft_time_past,
            "future_time_features" : ft_time_future

        }
        
        return ret