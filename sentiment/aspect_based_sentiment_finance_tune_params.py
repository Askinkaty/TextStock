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
import transformers
from ray.air.integrations.wandb import WandbLoggerCallback
import ray

import mlflow
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch  # Import OptunaSearch

from ray.tune import CLIReporter
import wandb

ray.init(num_gpus=2)
from accelerate import Accelerator
import logging



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
    table = table.sample(frac=1, random_state=42).reset_index(drop=True) # Added random_state for reproducibility
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
    train_df = train_df[:3000]
    # Despite the dataset being balanced, make the number of classes equal
    sentiment_counts = train_df['sentiment'].value_counts()
    min_count = sentiment_counts.min()
    balanced_train = (
        train_df.groupby('sentiment', group_keys=False)
        .apply(lambda x: x.iloc[:min_count])
    )
    big_test = train_df[3000:]
    val_df = val_df[:300]
    test_df = test_df[:300]
    logging.info(f"Data split: {len(balanced_train)} training, {len(val_df)} validation, {len(test_df)} test samples. "
                 f"Big test: {len(big_test)}")
    return train_df, val_df, test_df, big_test


class FinDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        ordinal = 0 # if label == 'positive' else 0 # issues with Trainer if ordinal=1; never learns positive label
        return {
            'text': row['Title'],
            'span': row['span'],
            'label': row['sentiment'],
            'ordinal': ordinal
        }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def get_data():
    data_dir = '/scratch/project_2006600/fin_experiment/data'
    table = pd.read_csv(os.path.join(data_dir, 'SEntFiN-v1.1.csv'))
    table = prepare_dataset(table)
    train_df, valid_df, test_df, big_test = split_data(table)
    big_test.to_csv(os.path.join(data_dir, 'big_test.csv'))
    logging.info('Saved big test file')
    train_data = FinDataset(train_df, max_len=15)
    valid_data = FinDataset(valid_df, max_len=15)
    test_data = FinDataset(test_df, max_len=15)
    big_test_data = FinDataset(big_test, max_len=15)
    return train_data, valid_data, test_data, big_test_data


def train_absa(config):
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = AbsaModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        spacy_model="en_core_web_sm",
    )
    model.to(device)

    output_dir = '/scratch/project_2006600/fin_experiment/models'
    train_data, valid_data, test_data, _ = get_data()

    wandb_project_name = "financial_sentiment_absa_pbt"  # Define your wandb project name

    with wandb.init(project=wandb_project_name): # Initialize wandb run
        args = TrainingArguments(
            output_dir=output_dir,
            num_epochs=2,
            use_amp=True,
            batch_size=64,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=100,
            load_best_model_at_end=True,
            report_to=None,
        )

        trainer = AbsaTrainer(
            model,
            args=args,
            train_dataset=train_data,
            eval_dataset=valid_data,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        train_result = trainer.train()
        train_metrics = train_result.metrics
        wandb.log(train_metrics) # Log training metrics to wandb

        evaluation_result = trainer.evaluate(test_data)
        eval_metrics = evaluation_result
        wandb.log(eval_metrics) # Log evaluation metrics to wandb

        # Report the validation metric to Ray Tune
        tune.report(
            eval_accuracy=evaluation_result.get("accuracy"),
        )
        # Save the trained model within the trial directory
        trainer.save_model(output_dir)


def main():
    config = {
        "num_epochs": tune.choice([2, 3]),
        "batch_size": tune.choice([32, 64]),
        "learning_rate": tune.loguniform(1e-5, 1e-4),
        "weight_decay": tune.uniform(0.0, 0.1),
    }

    reporter = CLIReporter(
        metric_columns=["accuracy", "training_iteration"]
    )

    wandb_callback = WandbLoggerCallback(
        project="financial_sentiment_absa_pbt"
    )

    # Initialize the Optuna search algorithm
    optuna_search = OptunaSearch(metric="accuracy", mode="max")

    tuner = tune.Tuner(
        train_absa,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,  # Specify the search algorithm
            num_samples=10,  # Number of trials
            resources_per_trial={"gpu": 0.5}  # Allows two trials per GPU
        ),
        run_config=train.RunConfig(
            callbacks=[wandb_callback],
            name="optuna_search_absa",
            progress_reporter=reporter
        ),
        param_space=config,
        # Ray will automatically create directories for each trial
    )

    results = tuner.fit()

    print("Best trial config: {}".format(results.get_best_result().config))
    print("Best trial validation metrics: {}".format(results.get_best_result().metrics))



if __name__ == '__main__':
    main()