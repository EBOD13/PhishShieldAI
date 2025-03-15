import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# from google.colab import drive # Uncomment when using mounting from Google Drive on Colab


# Load the combined dataset
# combined_data = pd.read_csv("/content/drive/MyDrive/datasets/combined_email_data.csv") # Uncomment when loading using Colab
combined_data = pd.read_csv("datasets/combined_email_data.csv")

# Ensure the text and label columns are clean
combined_data['text'] = combined_data['text'].astype(str).fillna('')
combined_data['label'] = combined_data['label'].astype(int)

# Split into training and validation sets
train_data, val_data = train_test_split(combined_data, test_size=0.2, random_state=42)

# Load the tokenizer
model_name = "distilbert-base-uncased"  # You can use "bert-base-uncased" or "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenize the text data
def tokenize_data(texts, labels, max_length=128):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ), torch.tensor(labels.tolist())


# Tokenize training and validation data
train_encodings, train_labels = tokenize_data(train_data['text'], train_data['label'])
val_encodings, val_labels = tokenize_data(val_data['text'], val_data['label'])


# Create a PyTorch Dataset
class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# Create datasets
train_dataset = EmailDataset(train_encodings, train_labels)
val_dataset = EmailDataset(val_encodings, val_labels)

# Load the Pre-trained Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model
model_save_path = "models/fine_tuned_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Evaluate the model
results = trainer.evaluate()
print("Validation results:", results)
