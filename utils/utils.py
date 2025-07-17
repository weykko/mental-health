import torch
import re
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


def preprocess_text(text):
    if not isinstance(text, str): return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user mentions
    text = re.sub(r'@\w+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def binary_accuracy(preds, y):
    # Используем torch.argmax для получения метки с максимальной вероятностью
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def save_model(model, tokenizer, output_dir):
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nModel saved to {output_dir}")


# Function to plot training and test results
def plot_class_distribution(class_counts, path):
    fig, ax = plt.subplots()

    ax.bar(class_counts.index.values, class_counts.values, color=['#f76452', '#58db6e'], width=0.8)

    ax.set_ylabel("Number of messages")
    ax.set_title("Class Distribution")

    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_training_results(history, path):
    epochs = len(history['train_losses'])

    # Create subplots for loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Train and Test Loss
    ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss')
    ax1.plot(range(1, epochs + 1), history['train_losses'], label='Train Loss', color='#97a6c4')
    ax1.plot(range(1, epochs + 1), history['test_losses'], label='Test Loss', color='#384860')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.set_xticks(range(1, epochs + 1))
    ax1.set_title("Loss", fontsize=14)

    # Create another y-axis for accuracy
    ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Accuracy')
    ax2.plot(range(1, epochs + 1), history['train_accs'], label='Train Accuracy', color='#97a6c4')
    ax2.plot(range(1, epochs + 1), history['test_accs'], label='Test Accuracy', color='#384860')
    ax2.legend(loc='lower right', fontsize=12)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.set_xticks(range(1, epochs + 1))
    ax2.set_title("Accuracy", fontsize=14)

    fig.tight_layout()
    plt.savefig(path)
    plt.show()