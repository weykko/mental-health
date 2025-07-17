from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import random

from utils.utils import preprocess_text, plot_class_distribution, save_model, plot_training_history
from utils.datasets import get_depression_loaders
from utils.trainer import train_model, evaluate_model


def feature_engineering(df):
    """
    Добавление новых признаков для улучшения модели (Feature Engineering).

    Параметры:
    df (pandas.DataFrame): Исходный DataFrame с текстами и метками.

    Возвращает:
    pandas.DataFrame: Обновленный DataFrame с новыми признаками.
    """

    # 1. Признаки на основе длины текста
    df['text_length'] = df['processed_text'].apply(len)
    df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))

    # 2. Признаки, указывающие на возможное наличие суицидальных мыслей
    suicide_indicators = [
        'worth', 'despair', 'broken', 'numb', 'hopeless', 'tragic', 'shame', 'rejected',
        'kill', 'pain', 'life', 'anymore', 'want', 'hope', 'die', 'suicide', 'end',
        'abandoned', 'isolated', 'miserable', 'helpless', 'desperate', 'grief', 'torment',
        'help', 'death', 'dead', 'hate', 'tired', 'pills', 'hurt', 'alone', 'sad',
        'depression', 'anxiety', 'lost', 'cut', 'empty', 'worthless', 'cry', 'suffer',
    ]

    # Для каждого индикатора создаем бинарный признак
    for word in suicide_indicators:
        df[f'has_{word}'] = df['processed_text'].apply(lambda x: 1 if word in x.split() else 0)

    # 3. Подсчет количества местоимений первого лица
    first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself']
    df['first_person_count'] = df['processed_text'].apply(
        lambda x: sum(1 for word in x.split() if word in first_person_pronouns)
    )

    return df


# Устанавливаем фиксированные значения случайных чисел для воспроизводимости
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Определяем устройство для вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Загрузка датасета из CSV в DataFrame
df = pd.read_csv('./datasets/suicide_detection.csv')
print(f"Размер датасета: {df.shape}")
print(df.head())

# Проверка распределения классов в датасете
print("Распределение классов в датасете:")
print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True) * 100)

# Визуализация распределения классов
class_counts = df['class'].value_counts()
plot_class_distribution(class_counts, "plots/class_distribution.png")

# Предобработка текста
df['processed_text'] = df['text'].apply(preprocess_text)

# Удаление строк с пропущенными текстами
df = df.dropna(subset=['processed_text'])
df = df[df['processed_text'] != ""]

# Преобразование меток классов в числовые значения
df['label'] = df['class'].map({'suicide': 1, 'non-suicide': 0})

# Feature Engineering
df = feature_engineering(df)

# Подготовка данных
# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Параметры для дата лоадера
max_length = 512
batch_size = 64

# BERT токенизатор
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Загрузка данных в DataLoader
train_loader, test_loader = get_depression_loaders(X_train, y_train, X_test, y_test, tokenizer, max_length, batch_size)

# Инициализация модели BERT от Hugging Face (библиотека transformers) для классификации последовательностей
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
).to(device)

# Определение оптимизатора
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3

# Создание планировщика с прогревом
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * epochs
)

# Обучение модели
model, history = train_model(model, tokenizer, train_loader, test_loader, optimizer, scheduler, device, epochs)

# Визуализация истории обучения
plot_training_history(history, "plots/train_history.png")

# Оценка модели на тестовой выборке
print(evaluate_model(model, test_loader, device))

# Сохранение модели и токенизатора
save_model(model, tokenizer, "/model/")
