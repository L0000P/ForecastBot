import glob
import pandas as pd
from models.MultivariateTSDataset import MultivariateTSDataset
from pathlib import Path
from transformers import (
    EarlyStoppingCallback,
    InformerConfig,
    InformerForPrediction,
    Trainer,
    TrainingArguments,
)
import numpy as np
from sklearn.metrics import mean_squared_error

log_path   = "Data/Log/"      # Log Path
model_path = "Data/Model/"    # Model Path    

Path(log_path).mkdir(parents=True, exist_ok=True) # Make Dir Log.
Path(model_path).mkdir(parents=True, exist_ok=True) # Make Dir Model.

context_length   = 1024 # Context Length.
forecast_horizon = 192 # Forecast Horizon.
patch_length     = 64 # Patch Length.
num_workers      = 8 # Number of Workers.  
batch_size       = 4 # Batch Size.

# Ask user for input
timestamp_column = input("Insert timestamp column name<|date|>:") or "date"
csv_files = glob.glob(input("Insert Path of csv files<|../data/ETTh1.csv|>: ")) or ["../data/ETTh1.csv"]
delimiter = input("Insert delimiter of dataset:<|,|>:") or ","
index_start = input("Insert start index dataset:<|1|>:") or 1
resume_from_checkpoint = input("Resume from checkpoint?(<y>,n):") or "y"

# Check if the user wants to resume from a checkpoint
if resume_from_checkpoint == "" or resume_from_checkpoint == "y" or resume_from_checkpoint == "yes":
    resume_from_checkpoint = True
if not(resume_from_checkpoint == "" or resume_from_checkpoint == "y" or resume_from_checkpoint == "yes"):
    resume_from_checkpoint = False

dataframes = [pd.read_csv(f, parse_dates=[timestamp_column], delimiter=delimiter) for f in csv_files] # Read CSV Files.
dataset = pd.concat(dataframes, ignore_index=True) # Concatenate DataFrames.
past_columns = [col for col in dataset.columns[index_start:]] # Forecast Columns.
forecast_columns = [col for col in dataset.columns[index_start:]] # Forecast Columns.

sample_number    = len(dataset)   # Sample Number 
num_train        = int(sample_number * 0.7) # Num Train
num_test         = int(sample_number * 0.2) # Num Test
num_valid        = sample_number - num_train - num_test # Num Valid

f1, e1           = 0, num_train # Start and End Train
f2, e2           = e1 - context_length, e1 + num_valid # Start and End Valid
f3, e3           = e2 - context_length, sample_number # Start and End Test

train_data = dataset.iloc[f1:e1, :].reset_index(drop=True) # Train Data
valid_data = dataset.iloc[f2:e2, :].reset_index(drop=True) # Valid Data
test_data  = dataset.iloc[f3:e3, :].reset_index(drop=True) # Test Data

train_dataset = MultivariateTSDataset( # Train Dataset
    train_data, # Train Data
    timestamp_column  = timestamp_column, # Time Stamp Column
    training_columns  = past_columns, # Training Columns
    target_columns    = forecast_columns, # Target Columns
    context_length    = context_length, # Context Length
    prediction_length = forecast_horizon, # Prediction Length
)

valid_dataset = MultivariateTSDataset(
    valid_data, #  Valid Data
    timestamp_column  = timestamp_column, # Time Stamp Column
    training_columns  = past_columns, # Past Columns
    target_columns    = forecast_columns, # Forecast Columns
    context_length    = context_length, # Context Length
    prediction_length = forecast_horizon, # Prediction Horizon
)

test_dataset  = MultivariateTSDataset( # Test Dataset
    test_data, # Test Data
    timestamp_column  = timestamp_column, # Time Stamp Column
    training_columns  = past_columns, # Training Columns
    target_columns    = forecast_columns, # Target Columns
    context_length    = context_length, # Context Length
    prediction_length = forecast_horizon, # Prediction Horizon
)

config = InformerConfig( # Informer Config
    prediction_length       = forecast_horizon, # Prediciton Length
    context_length          = context_length, # Context Length
    input_size              = len(past_columns), # Input Size
    scaling                 = "std", # Scaling
    d_model                 = 128, # D Model
    encoder_layers          = 2, # Encoder Layers
    decoder_layers          = 3, # Decoder Layers
    encoder_attention_heads = 8, # Encoder Attention Heads
    decoder_attention_heads = 8, # Decoder Attention Heads
    encoder_ffn_dim         = 256, # Encoder FFN Dim
    decoder_ffn_dim         = 256, # Decoder FFN Dim
    dropout                 = 0.2, # Dropout
    encoder_layerdrop       = 0.2, # Encoder Layer Drop
    decoder_layerdrop       = 0.2, # Decoder Layer Drop
    num_parallel_samples    = 1, # Num Parallel Samples
    attention_type          = "prob", # Attention Type
    distil                  = True, # Distil
    lags_sequence           = [ 0 ], # Lags Sequence
    num_time_features       =  8 # Num Time Features
)

device = "cuda" #   Device
model = InformerForPrediction(config).to(device) # Model


training_args  = TrainingArguments( # Training Arguments
    output_dir                  = model_path, # Output Dir
    overwrite_output_dir        = True, # Overwrite Output Dir
    num_train_epochs            = 200, # Num Train Epochs
    do_eval                     = True, # Do Eval
    eval_strategy               = "epoch", # Evaluation Strategy
    per_device_train_batch_size = batch_size, # Per Device Train Batch Size
    per_device_eval_batch_size  = batch_size, # Per Device Eval Batch Size
    save_strategy               = "epoch", # Save Strategy
    logging_strategy            = "epoch", # Logging Strategy
    save_total_limit            = 3, # Save Total Limit
    logging_dir                 = log_path, # Logging Dir
    load_best_model_at_end      = True, # Load Best Model At End
    metric_for_best_model       = "eval_loss", # Metric For Best Model
    greater_is_better           = False, # Greater Is Better
    label_names                 = ["future_values"], # Label Names
    report_to                   = "none" # Report To
)

early_stopping_callback = EarlyStoppingCallback( # Early Stopping Callback
    early_stopping_patience  = 10, # Early Stopping Patience
    early_stopping_threshold = 0.0001 # Early Stopping Threshold
)

trainer = Trainer( # Trainer
    model         = model, # Model
    args          = training_args, # Training Arguments
    train_dataset = train_dataset, # Train Dataset
    eval_dataset  = valid_dataset, # Eval Dataset
    callbacks     = [early_stopping_callback] # Callbacks
)

trainer.train(resume_from_checkpoint=resume_from_checkpoint) # Train

# Make predictions on the test dataset
print("Generating predictions on the test dataset...")
test_predictions = trainer.predict(test_dataset)

# Extract the predicted and actual values
y_true = np.array([example['future_values'] for example in test_dataset])
y_pred = test_predictions.predictions

# Calculate the MSE
mse = mean_squared_error(y_true, y_pred)

# Print the MSE
print(f"Mean Squared Error on the test dataset: {mse:.4f}")