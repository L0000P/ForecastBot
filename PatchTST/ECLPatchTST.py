import os
import pandas as pd
import warnings
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    set_seed,
    Trainer,
    TrainingArguments,
)


warnings.filterwarnings("ignore", module="torch") # Ignore Torch Warnings.
set_seed(2023) # Set Seed for Reproducibility.

dataset_path = "../data/ECL.csv" # Path Dataset.
timestamp_column = "date" # Column Name Timestamp.
id_columns = [] # Column Names for the IDs.

context_length = 512    # Length Context Window.
forecast_horizon = 96   # Forecast Horizon.
patch_length = 16   # Length Extracted from Context Window.
num_workers = 16    # Workers CPU.
batch_size = 4  # GPU Memory.

data = pd.read_csv(dataset_path, parse_dates=[timestamp_column]) # Load Dataset.
forecast_columns = list(data.columns[1:]) # Get Forecast Columns.

num_train = int(len(data) * 0.7) # 70% Data Training.
num_test = int(len(data) * 0.2) # 20% Data Testing.
num_valid = len(data) - num_train - num_test # 10% Data Validation.

# 1st Stride Indices.
border1s = [ 0, num_train - context_length, len(data) - num_test - context_length]

# 2nd Stride Indices.
border2s = [num_train, num_train + num_valid, len(data)]

train_start_index = border1s[0] # Start Index Training.
train_end_index = border2s[0] # End Index Training.

valid_start_index = border1s[1] # Start Index Validation.
valid_end_index = border2s[1] # End Index Validation.

test_start_index = border1s[2] # Start Index Testing.
test_end_index = border2s[2] # End Index Testing.

# Test Data.
train_data = select_by_index(data, id_columns=id_columns, start_index=train_start_index, end_index=train_end_index)

# Validation Data.
valid_data = select_by_index(data,id_columns=id_columns,start_index=valid_start_index,end_index=valid_end_index)

# Test Data.
test_data = select_by_index(data,id_columns=id_columns,start_index=test_start_index,end_index=test_end_index)

# Time Series Preprocessor.
time_series_preprocessor = TimeSeriesPreprocessor(timestamp_coumn=timestamp_column,id_columns=id_columns,input_column=forecast_columns,output_colums=forecast_columns,scaling=True)

# Preproocess Data.
time_series_preprocessor = time_series_preprocessor.train(train_data)

# Train Dataset.
train_dataset = ForecastDFDataset(
    time_series_preprocessor.preprocess(train_data), # Preprocess Data.
    id_columns=id_columns, # ID Columns.
    timestamp_column="date", # Timestamp Column.
    observable_columns=forecast_columns, # Observable Columns.
    target_columns=forecast_columns, # Target Columns.
    context_length=context_length, # Context Length.
    prediction_length=forecast_horizon # Prediction Length.
)

# Validation Dataset.
valid_dataset = ForecastDFDataset(
    time_series_preprocessor.preprocess(valid_data), # Preprocess Data.
    id_columns=id_columns, # ID Columns.
    timestamp_column="date", # Timestamp Column.
    observable_columns=forecast_columns, # Observable Columns.
    target_columns=forecast_columns, # Target Columns.
    context_length=context_length, # Context Length.
    prediction_length=forecast_horizon # Prediction Length.
)

# Test Dataset.
test_dataset = ForecastDFDataset(
    time_series_preprocessor.preprocess(test_data), # Preprocess Data.
    id_columns=id_columns, # ID Columns.
    timestamp_column="date", # Timestamp Column.
    observable_columns=forecast_columns, # Observable Columns.
    target_columns=forecast_columns, # Target Columns.
    context_length=context_length, # Context Length.
    prediction_length=forecast_horizon # Prediction Length.
)

# Config PatchTST Model.
config = PatchTSTConfig(
    num_input_channels=len(forecast_columns), # Number Input Channels.
    context_length=context_length, # Context Length.
    patch_length=patch_length, # Patch Length.
    patch_stride=patch_length, # Patch Stride.
    prediction_length=forecast_horizon, # Prediction Length.
    random_mask_ratio=0.4, # Random Mask Ratio.
    d_model=128, # Model Dimension.
    num_attention_heads=16, # Number Attention Heads.
    num_hidden_layers=3, # Number Hidden Layers.
    ffn_dim=256, # Feed Forward Dimension.
    dropout=0.2, # Dropout.
    head_dropout=0.2, # Head Dropout.
    pooling_type=None, # Pooling Type.
    channel_attention=False, # Channel Attention.
    scaling="std", # Scaling.
    loss="mse", # Loss.
    pre_norm=True, # Pre Norm.
    norm_type="batchnorm" # Norm Type.
)

# Train Model.
model = PatchTSTForPrediction(config)

# Training Arguments.
training_args = TrainingArguments(
    output_dir="./checkpoint/patchtst/electricity/pretrain/output/", # Output Directory.    
    overwrite_output_dir=True, # Overwrite Output Directory.
    num_train_epochs=100, # Number Training Epochs.
    do_eval=True, # Do Evaluation.
    evaluation_strategy="epoch", # Evaluation Strategy. 
    per_device_train_batch_size=batch_size, # Batch Size.     
    per_device_eval_batch_size=batch_size, # Evaluation Batch Size.
    dataloader_num_workers=num_workers, # Number Workers CPU.
    save_strategy="epoch", # Save Strategy.
    logging_strategy="epoch", # Logging Strategy.
    save_total_limit=3, # Save Total Limit.
    logging_dir="./checkpoint/patchtst/electricity/pretrain/logs/", # Logging Directory.  
    load_best_model_at_end=True,  # Load the best model when training ends.
    metric_for_best_model="eval_loss",  # Metric for Best Model.
    greater_is_better=False,  # Greater is Better.
    label_names=["future_values"], # Label Names.
)

# Create Early Stopping Callback.
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10, # Early Stopping Patience. 
    early_stopping_threshold=0.0001   # Early Stopping Threshold.
)

# Define Trainer.
trainer = Trainer(
    model=model, # Model.
    args=training_args, # Training Arguments.
    train_dataset=train_dataset, # Train Dataset.
    eval_dataset=valid_dataset, # Validation Dataset.
    callbacks=[early_stopping_callback] # Callbacks.
)

# Train Model.
trainer.train() 

# Results.
results = trainer.evaluate(test_dataset)
print("Test result:")
print(results)

# Save Model.
save_dir = "patchtst/electricity/model/pretrain/"
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)