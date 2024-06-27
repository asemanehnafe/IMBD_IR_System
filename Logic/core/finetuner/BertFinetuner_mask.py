from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import json
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from huggingface_hub import login, create_repo


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.dataset = None
        self.top_genres = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=top_n_genres)
        self.train_texts = None
        self.train_labels = None
        self.val_labels = None
        self.val_texts = None
        self.test_texts = None
        self.test_labels = None
        


    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        self.dataset = pd.DataFrame(data)
        self.dataset = self.dataset[['first_page_summary', 'genres']]
        # Remove rows where 'first_page_summary' or 'genres' are None
        self.dataset.dropna(subset=['first_page_summary', 'genres'], inplace=True)

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        # Implement genre filtering and visualization logic
        genre_counter = Counter([genre for sublist in self.dataset['genres'] for genre in sublist])
        self.top_genres = {genre for genre, _ in genre_counter.most_common(self.top_n_genres)}

        def filter_genres(genres):
            return [genre if genre in self.top_genres else 'Other' for genre in genres]

        self.dataset['genres'] = self.dataset['genres'].apply(filter_genres)
        self.dataset = self.dataset[self.dataset['genres'].apply(lambda x: 'Other' not in x)]
        self.dataset = self.dataset[self.dataset['genres'].apply(lambda x: x != [])]
        self.dataset['genres'] = self.dataset['genres'].apply(lambda x: x[0])  # Take the first genre if multiple
        self.dataset = self.dataset[self.dataset['genres'] != 'Other']
        genre_encoder = {genre: idx for idx, genre in enumerate(self.top_genres)}
        self.dataset['genres'] = self.dataset['genres'].map(genre_encoder)

        top_genre_counts = {genre: genre_counter[genre] for genre in self.top_genres}
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(top_genre_counts.keys()), y=list(top_genre_counts.values()))
        plt.title('Top Genres Distribution')
        plt.xlabel('Genres')
        plt.ylabel('Frequency')
        plt.show()

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        # Implement dataset splitting logic
        train_val_data, test_data = train_test_split(self.dataset, test_size=test_size, stratify=self.dataset['genres'])
        train_data, val_data = train_test_split(train_val_data, test_size=val_size, stratify=train_val_data['genres'])

        self.train_texts, self.train_labels = list(train_data['first_page_summary']), list(train_data['genres'])
        self.val_texts, self.val_labels = list(val_data['first_page_summary']), list(val_data['genres'])
        self.test_texts, self.test_labels = list(test_data['first_page_summary']), list(test_data['genres'])


    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        # Implement dataset creation logic
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        # Implement BERT fine-tuning logic
        train_encodings = self.tokenizer(self.train_texts, truncation=True, padding=True)
        val_encodings = self.tokenizer(self.val_texts, truncation=True, padding=True)

        train_dataset = self.create_dataset(train_encodings, self.train_labels)
        val_dataset = self.create_dataset(val_encodings, self.val_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        self.model = trainer

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
          pred (EvalPrediction): The model's predictions.

        Returns:
          dict: A dictionary containing the computed metrics.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, F1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        return {
          'accuracy': accuracy_score(labels, preds),
          'f1': F1,
          'precision': precision,
          'recall': recall
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        # Implement model evaluation logic
        test_encodings = self.tokenizer(self.test_texts, truncation=True, padding=True)
        test_dataset = self.create_dataset(test_encodings, self.test_labels)
        predictions, _, _ = self.model.predict(test_dataset)
        y_pred = np.argmax(predictions, axis=1)
        precision, recall, F1, _ = precision_recall_fscore_support(self.test_labels, y_pred, average='weighted', zero_division=0)
        results = {
            'accuracy': accuracy_score(self.test_labels, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': F1
        }
        print(results)
        return results

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        # Implement model saving logic
#         self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)

        token = "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"
        login(token)
#         repo_url = create_repo(repo_id=model_name)
        self.model.push_to_hub(model_name, token)
        self.tokenizer.push_to_hub(model_name,token)

class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        # TODO: Implement initialization logic
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        # Implement item retrieval logic
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)
    

