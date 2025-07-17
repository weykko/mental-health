import torch
import re
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def preprocess_text(text):
    """
    Предобработка текста перед подачей в модель.

    Параметры:
    text (str): Входной текст для обработки.

    Возвращает:
    str: Предобработанный текст.
    """
    if not isinstance(text, str): return ""

    # Преобразуем в нижний регистр
    text = text.lower()

    # Удаляем URL-адреса
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Удаляем упоминания пользователей (например, @username)
    text = re.sub(r'@\w+', '', text)

    # Удаляем специальные символы и цифры
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def binary_accuracy(preds, y):
    """
    Вычисление точности для бинарной классификации.

    Параметры:
    preds (Tensor): Прогнозы модели (логиты).
    y (Tensor): Истинные метки классов.

    Возвращает:
    float: Точность классификации.
    """
    # Используем torch.argmax для получения метки с максимальной вероятностью
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()  # Сравниваем предсказания с истинными метками
    acc = correct.sum() / len(correct)  # Средняя точность

    return acc


def save_model(model, tokenizer, output_dir):
    """
    Сохранение модели и токенизатора.

    Параметры:
    model (torch.nn.Module): Модель, которую нужно сохранить.
    tokenizer (PreTrainedTokenizer): Токенизатор для сохранения.
    output_dir (str): Путь к директории, куда будет сохранена модель.

    Возвращает:
    None
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nModel saved to {output_dir}")


def plot_class_distribution(class_counts, path):
    """
    Построение графика распределения классов.

    Параметры:
    class_counts (pandas.Series): Количество сообщений по классам.
    path (str): Путь для сохранения изображения.

    Возвращает:
    None
    """
    fig, ax = plt.subplots()

    ax.bar(class_counts.index.values, class_counts.values, color=['#f76452', '#58db6e'], width=0.8)

    ax.set_ylabel("Number of messages")
    ax.set_title("Class Distribution")

    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_training_history(history, path):
    """
    Построение графиков для потерь и точности в процессе обучения.

    Параметры:
    history (dict): История обучения, содержащая потери и точность для каждой эпохи.
    path (str): Путь для сохранения изображения.

    Возвращает:
    None
    """
    epochs = len(history['train_losses'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.set_xlabel('Epoch')
    ax1.plot(range(1, epochs + 1), history['train_losses'], label='Train Loss', color='#97a6c4')
    ax1.plot(range(1, epochs + 1), history['test_losses'], label='Test Loss', color='#384860')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.set_xticks(range(1, epochs + 1))
    ax1.set_title("Loss", fontsize=14)

    ax2.set_xlabel('Epoch')
    ax2.plot(range(1, epochs + 1), history['train_accs'], label='Train Accuracy', color='#97a6c4')
    ax2.plot(range(1, epochs + 1), history['test_accs'], label='Test Accuracy', color='#384860')
    ax2.legend(loc='lower right', fontsize=12)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.set_xticks(range(1, epochs + 1))
    ax2.set_title("Accuracy", fontsize=14)

    fig.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, path):
    """
    Построение матрицы ошибок (confusion matrix) и визуализация её с помощью тепловой карты.

    Параметры:
    all_labels (list or array-like): Истинные метки классов.
    all_preds (list or array-like): Предсказанные метки классов.
    path (str): Путь для сохранения изображения.

    Возвращает:
    None
    """
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=['non-suicidal', 'suicidal'],
                yticklabels=['non-suicidal', 'suicidal'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
