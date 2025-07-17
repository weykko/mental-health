from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import random

from utils.utils import preprocess_text, plot_class_distribution, save_model
from utils.datasets import get_depression_loaders
from utils.trainer import train_and_evaluate_model

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Загрузка датасета из csv в DataFrame
df = pd.read_csv('./datasets/suicide_detection.csv')
print(f"Размер датасета: {df.shape}")
print(df.head())

# Check class distribution
print("\Распределение классов в датасете:")
print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True) * 100)

class_counts = df['class'].value_counts()
plot_class_distribution(class_counts, "plots/class_distribution.png")

# Предобработка текста
df['processed_text'] = df['text'].apply(preprocess_text)

# Remove rows with missing text
df = df.dropna(subset=['processed_text'])
df = df[df['processed_text'] != ""]

# Map class labels to numeric values
df['label'] = df['class'].map({'suicide': 1, 'non-suicide': 0})


def feature_engineering(df):
    # Feature Engineering

    # 1. Text length features
    df['text_length'] = df['processed_text'].apply(len)
    df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))

    # 2. Identify potential suicide-related keywords
    suicide_indicators = [
        'worth', 'despair', 'broken', 'numb', 'hopeless', 'tragic', 'shame', 'rejected',
        'kill', 'pain', 'life', 'anymore', 'want', 'hope', 'die', 'suicide', 'end',
        'abandoned', 'isolated', 'miserable', 'helpless', 'desperate', 'grief', 'torment',
        'help', 'death', 'dead', 'hate', 'tired', 'pills', 'hurt', 'alone', 'sad',
        'depression', 'anxiety', 'lost', 'cut', 'empty', 'worthless', 'cry', 'suffer',
    ]

    for word in suicide_indicators:
        df[f'has_{word}'] = df['processed_text'].apply(lambda x: 1 if word in x.split() else 0)

    # 3. Count first-person pronouns
    first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself']
    df['first_person_count'] = df['processed_text'].apply(
        lambda x: sum(1 for word in x.split() if word in first_person_pronouns)
    )

    return df


df = feature_engineering(df)

# Prepare data for BERT
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

max_length = 512
batch_size = 64
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_loader, test_loader = get_depression_loaders(X_train, y_train, X_test, y_test, tokenizer, max_length, batch_size)

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3

# Create scheduler with warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * epochs
)

model = train_and_evaluate_model(model, tokenizer, train_loader, test_loader, optimizer, scheduler, device, epochs)
save_model(model, tokenizer, "/model/")
