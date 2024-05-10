import os
import pandas as pd
import warnings
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index
from transformers import (
    EarlyStoppingCallback,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
    set_seed
)

warnings.filterwarnings("ignore", module="torch") # Ignore Torch Warnings.
set_seed(2023)  # Set seed for reproducibility.

dataset = "ETTh1" # Dataset Name.

context_length = 512    # Amount of historical data used as input to the model.
forecast_horizon = 96   # Number of timestamps to forecast in the future.
patch_length = 16   # Patch length for the PatchTST model.
num_workers = 16 # Number of CPU cores to use for data preprocessing.
batch_size = 4 # According to GPU Memory.


print(f"Loading target dataset: {dataset}") # Load Dataset.
dataset_path = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset}.csv" # Dataset Path.
timestamp_column = "date" # Column Name Timestamp.
id_columns = [] # Column Names for the IDs.
forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"] # Forecast Columns.
train_start_index = None # Start Index Training.
train_end_index = 12 * 30 * 24 # End Index Training.

valid_start_index = 12 * 30 * 24 - context_length # Start Index Validation.
valid_end_index = 12 * 30 * 24 + 4 * 30 * 24 # End Index Validation.

test_start_index = 12 * 30 * 24 + 4 * 30 * 24 - context_length # Start Index Testing.
test_end_index = 12 * 30 * 24 + 8 * 30 * 24 # End Index Testing.

# Load Dataset.
data = pd.read_csv(dataset_path,parse_dates=[timestamp_column])

# Train Data.
train_data = select_by_index(data,id_columns=id_columns,start_index=train_start_index,end_index=train_end_index)

# Valid Data.
valid_data = select_by_index(data,id_columns=id_columns,start_index=valid_start_index,end_index=valid_end_index)

test_data = select_by_index(data,id_columns=id_columns,start_index=test_start_index,end_index=test_end_index)

# Time Series Preprocessor.
time_series_preprocessor = TimeSeriesPreprocessor(timestamp_column=timestamp_column,id_columns=id_columns,input_columns=forecast_columns,output_columns=forecast_columns,scaling=True)

# Train Data.
time_series_preprocessor = time_series_preprocessor.train(train_data)

# Train Dataset.
train_dataset = ForecastDFDataset(
    time_series_preprocessor.preprocess(train_data), # Preprocess Data.
    id_columns=id_columns, # ID Columns.
    observable_columns=forecast_columns, # Observable Columns.
    target_columns=forecast_columns, # Target Columns.
    context_length=context_length, # Context Length.
    prediction_length=forecast_horizon # Prediction Length.
)

# Valid Dataset.
valid_dataset = ForecastDFDataset(
    time_series_preprocessor.preprocess(valid_data), # Preprocess Data.
    id_columns=id_columns, # ID Columns.
    observable_columns=forecast_columns, # Observable Columns.
    target_columns=forecast_columns, # Target Columns.
    context_length=context_length, # Context Length.
    prediction_length=forecast_horizon # Prediction Length.
)

# Test Dataset.
test_dataset = ForecastDFDataset(
    time_series_preprocessor.preprocess(test_data), # Preprocess Data.
    id_columns=id_columns, # ID Columns.
    observable_columns=forecast_columns, # Observable Columns.
    target_columns=forecast_columns, # Target Columns.
    context_length=context_length, # Context Length.
    prediction_length=forecast_horizon # Prediction Length.
)

# Fine-Tune Forecast Model.
finetune_forecast_model = PatchTSTForPrediction.from_pretrained(
    "patchtst/electricity/model/pretrain/", # Pretrained Model.
    num_input_channels=len(forecast_columns), # Number of Input Channels.
    head_dropout=0.7 # Head Dropout. 
)

# Fine-Tune Forecast Arguments.
finetune_forecast_args = TrainingArguments(
    output_dir="./checkpoint/patchtst/transfer/finetune/output/", # Output Dir.
    overwrite_output_dir=True, # Overwrite Output Dir.
    learning_rate=0.0001, # Learning Rate.
    num_train_epochs=100, # Number of Training Epochs.
    do_eval=True, # Do Evaluation.
    evaluation_strategy="epoch", # Evaluation Strategy.
    per_device_train_batch_size=batch_size, # Batch Size.
    per_device_eval_batch_size=batch_size, # Batch Size.
    dataloader_num_workers=num_workers, # Number Workers CPU.
    report_to="tensorboard", # Report to Tensorboard.
    save_strategy="epoch", # Save Strategy.
    logging_strategy="epoch", # Logging Strategy.
    save_total_limit=3, # Save Total Limit.
    logging_dir="./checkpoint/patchtst/transfer/finetune/logs/", # Logging Dir.
    load_best_model_at_end=True,  # Load Best Model at End.
    metric_for_best_model="eval_loss",  # Metric for Best Model.
    greater_is_better=False, # Greater is Better.
    label_names=["future_values"], # Label Names.
)

# Early Stopping Callback.
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,  # Early Stopping Patience.
    early_stopping_threshold=0.001,  # Early Stopping Threshold.
)

# Fine Tune Forecast Trainer.
finetune_forecast_trainer = Trainer(
    model=finetune_forecast_model, # Model.
    args=finetune_forecast_args, # Arguments.
    train_dataset=train_dataset, # Train Dataset.
    eval_dataset=valid_dataset, # Eval Dataset.
    callbacks=[early_stopping_callback] # Callbacks.
)

# Results.
print("\n\nDoing zero-shot forecasting on target data")
result = finetune_forecast_trainer.evaluate(test_dataset)
print("Target data zero-shot forecasting result:")
print(result)

# Freeze Backbone.
for param in finetune_forecast_trainer.model.model.parameters():
    param.requires_grad = False

# Train Linear Probing.
print("\n\nLinear probing on the target data")
finetune_forecast_trainer.train()
print("Evaluating")
result = finetune_forecast_trainer.evaluate(test_dataset)
print("Target data head/linear probing result:")
print(result)

# Save Model.
save_dir = f"patchtst/electricity/model/transfer/{dataset}/model/linear_probe/"
os.makedirs(save_dir, exist_ok=True)
finetune_forecast_trainer.save_model(save_dir)

# Save Preprocessor.
save_dir = f"patchtst/electricity/model/transfer/{dataset}/preprocessor/"
os.makedirs(save_dir, exist_ok=True)
time_series_preprocessor.save_pretrained(save_dir)

# Reload Model.
finetune_forecast_model = PatchTSTForPrediction.from_pretrained(
    "patchtst/electricity/model/pretrain/", # Pretrained Model.
    num_input_channels=len(forecast_columns), # Number of Input Channels.
    dropout=0.7, # Dropout.
    head_dropout=0.7 # Head Dropout.
)

# Fine-Tune Forecast Arguments.
finetune_forecast_trainer = Trainer(
    model=finetune_forecast_model, # Model.
    args=finetune_forecast_args, # Arguments.
    train_dataset=train_dataset, # Train Dataset.
    eval_dataset=valid_dataset, # Eval Dataset.
    callbacks=[early_stopping_callback], # Callbacks.
)


print("\n\nFinetuning on the target data")
finetune_forecast_trainer.train() # Train.
print("Evaluating")
result = finetune_forecast_trainer.evaluate(test_dataset) # Evaluate.
print("Target data full finetune result:")
print(result)

# Save Model.
save_dir = f"patchtst/electricity/model/transfer/{dataset}/model/fine_tuning/"
os.makedirs(save_dir, exist_ok=True)
finetune_forecast_trainer.save_model(save_dir)
