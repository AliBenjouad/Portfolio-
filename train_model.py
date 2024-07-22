import os
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data):
        logger.info("Initializing the CustomDataset.")
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.encodings = []
        self.labels = []

        # Group data by sentence for tokenization
        try:
            grouped = data.groupby("sentence").apply(lambda x: x.reset_index(drop=True)).groupby(level=0, as_index=False)
            logger.info("Data grouped by sentence successfully.")
        except Exception as e:
            logger.error(f"Error grouping data by sentence: {e}")
            raise

        for sentence, group in grouped:
            words = group['token'].tolist()
            labels = group['label'].tolist()
            try:
                logger.info(f"Tokenizing sentence: {sentence}")
                logger.debug(f"Words: {words}")

                # Ensure words are in the correct format
                if not all(isinstance(word, str) for word in words):
                    logger.error(f"Invalid format for words in sentence: {sentence}. Words: {words}")
                    continue

                tokenized_inputs = self.tokenizer(words, is_split_into_words=True,
                                                  return_offsets_mapping=True, padding='max_length',
                                                  truncation=True, max_length=128)
                logger.info(f"Tokenized sentence: {sentence}")
            except Exception as e:
                logger.error(f"Error tokenizing sentence {sentence}: {e}")
                continue

            # Align labels
            try:
                word_ids = tokenized_inputs.word_ids(batch_index=0)
                label_ids = [-100 if word_id is None else labels[word_id] for word_id in word_ids]
                tokenized_inputs['labels'] = label_ids
                self.encodings.append(tokenized_inputs)
                logger.info(f"Aligned labels for sentence: {sentence}")
            except Exception as e:
                logger.error(f"Error aligning labels for sentence {sentence}: {e}")
                continue

        logger.info("CustomDataset initialized successfully.")

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        return item

    def __len__(self):
        return len(self.encodings)

def compute_metrics(pred):
    logger.info("Computing metrics.")
    try:
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        true_labels = [label for sublist in labels for label in sublist if label != -100]
        true_preds = [pred for sublist, label_sublist in zip(preds, labels)
                      for pred, label in zip(sublist, label_sublist) if label != -100]

        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average='binary', zero_division=0)
        acc = accuracy_score(true_labels, true_preds)
        logger.info("Metrics computed successfully.")
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        raise

def train_model():
    logger.info("Starting model training.")

    train_data_path = 'model/data/preprocessed_train_data.csv'
    valid_data_path = 'model/data/preprocessed_valid_data.csv'
    test_data_path = 'model/data/preprocessed_test_data.csv'

    try:
        logger.info(f"Loading training data from {train_data_path}.")
        train_data = pd.read_csv(train_data_path)
        logger.info(f"Loading validation data from {valid_data_path}.")
        valid_data = pd.read_csv(valid_data_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    logger.info("Creating datasets.")
    try:
        train_dataset = CustomDataset(train_data)
        valid_dataset = CustomDataset(valid_data)
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise

    logger.info("Loading the BERT model for token classification.")
    try:
        model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)
    except Exception as e:
        logger.error(f"Error loading BERT model: {e}")
        raise

    training_args = TrainingArguments(
        output_dir='./model/trained_models/results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./model/trained_models/logs',
        load_best_model_at_end=True
    )

    logger.info("Initializing the Trainer.")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics
        )
    except Exception as e:
        logger.error(f"Error initializing Trainer: {e}")
        raise

    logger.info("Starting training.")
    try:
        trainer.train()
        logger.info("Training completed.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

    logger.info("Saving the model.")
    try:
        model.save_pretrained('./model/trained_models/bert_model')
        train_dataset.tokenizer.save_pretrained('./model/trained_models')
        logger.info("Model and tokenizer saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model and tokenizer: {e}")
        raise

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        raise
