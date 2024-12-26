from setfit import AbsaModel
import pandas as pd
import os
import numpy as np
import copy
from torch.utils.data import Dataset
from setfit import AbsaTrainer, TrainingArguments
from transformers import EarlyStoppingCallback
import random
import torch
import mlflow

from accelerate import Accelerator
import logging

# Initialize the Accelerator
accelerator = Accelerator()

# Configure logging
logging.basicConfig(
    level=logging.INFO if accelerator.is_main_process else logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Suppress logging for non-main processes
if not accelerator.is_main_process:
    logging.getLogger().setLevel(logging.ERROR)


def prepare_dataset(table):
    """
    Prepares the dataset by expanding rows based on the 'Decisions' column.
    Each decision (aspect-sentiment pair) in the 'Decisions' dictionary becomes a separate row.

    Args:
        table (pd.DataFrame): Input DataFrame with a 'Decisions' column containing dictionaries of aspect-sentiment pairs.

    Returns:
        pd.DataFrame: A new DataFrame where each row represents a single aspect-sentiment pair.
    """
    rows = []
    table = table.sample(frac=1, random_state=42).reset_index(drop=True)  # Added random_state for reproducibility
    for i, row in table.iterrows():
        try:
            decisions = eval(row['Decisions'])
            if isinstance(decisions, dict):
                for k, v in decisions.items():
                    new_row = copy.deepcopy(row)
                    new_row['span'] = k
                    new_row['sentiment'] = v
                    rows.append(new_row)
            else:
                logging.warning(f"Skipping row {i} due to invalid 'Decisions' format: {row['Decisions']}")
        except Exception as e:
            logging.error(f"Error processing row {i}: {e}")
            logging.error(f"Problematic 'Decisions' value: {row['Decisions']}")

    new_table = pd.DataFrame(rows)
    logging.info(f"Dataset prepared with {len(new_table)} samples.")
    return new_table


def split_data(table):
    """Data should be shuffled already"""
    train_ratio = 0.8
    split_index = int(len(table) * train_ratio)
    train_df = table[:split_index]
    validation_all = table[split_index:]

    split_index_test = int(len(validation_all) * 0.5)
    val_df = validation_all[:split_index_test]
    test_df = validation_all[split_index_test:]
    # Limit the number of training examples. Train() gets stuck otherwise
    train_df = train_df[:2000]
    val_df = val_df[:300]
    test_df = test_df[:300]

    logging.info(f"Data split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples.")
    return train_df, val_df, test_df


class FinDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        span, label = row['Decisions']
        ordinal = 0
        return {
            'text': row['Title'],
            'span': span,
            'label': label,
            'ordinal': ordinal
        }


def main():
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = AbsaModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        spacy_model="en_core_web_sm",
    )
    model.to(device)

    model_dir = '/scratch/project_2006600/fin_experiment/models'
    data_dir = '/scratch/project_2006600/fin_experiment/data'
    output_dir = '/scratch/project_2006600/fin_experiment/models'

    table = pd.read_csv(os.path.join(data_dir, 'SEntFiN-v1.1.csv'))
    table = prepare_dataset(table)
    train_df, valid_df, test_df = split_data(table)

    train_data = FinDataset(train_df, max_len=15)
    valid_data = FinDataset(valid_df, max_len=15)
    test_data = FinDataset(test_df, max_len=15)

    mlflow_experiment_name = "financial_sentiment_absa"
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run():
        args = TrainingArguments(
            output_dir=output_dir,
            num_epochs=2,
            use_amp=True,
            batch_size=64,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=100,
            load_best_model_at_end=True,
            report_to='mlflow',
        )

        # Log parameters
        mlflow.log_param("pretrained_model_name_1", "sentence-transformers/all-MiniLM-L6-v2")
        mlflow.log_param("pretrained_model_name_2", "sentence-transformers/all-mpnet-base-v2")
        mlflow.log_param("spacy_model", "en_core_web_sm")
        mlflow.log_param("num_epochs", args.num_epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("eval_strategy", args.eval_strategy)
        mlflow.log_param("eval_steps", args.eval_steps)
        mlflow.log_param("save_steps", args.save_steps)
        mlflow.log_param("seed", args.seed)

        trainer = AbsaTrainer(
            model,
            args=args,
            train_dataset=train_data,
            eval_dataset=valid_data,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logging.info("Starting model training...")
        train_result = trainer.train()
        logging.info("Model training finished.")
        if train_result is not None and train_result.metrics:
            mlflow.log_metrics(train_result.metrics)

        try:
            model.save_pretrained(os.path.join(model_dir, "setfit-finance"))
            mlflow.log_artifact(os.path.join(model_dir, "setfit-finance"), artifact_path="trained_model")
            logging.info(f"Trained model saved to {os.path.join(model_dir, 'setfit-finance')}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

        logging.info("Evaluating the model on the test set...")
        test_result = trainer.evaluate(test_data)
        logging.info(f"Evaluation results: {test_result}")


if __name__ == '__main__':
    main()
