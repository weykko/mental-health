import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from utils.utils import binary_accuracy, save_model, plot_confusion_matrix


def train_epoch(model, train_loader, optimizer, scheduler, device):
    """
    Обучение модели в эпоху.

    Параметры:
    model (transformers.models.bert): Модель для обучения.
    train_loader (DataLoader): Загрузчик данных для обучения.
    optimizer (torch.optim.Optimizer): Оптимизатор для обновления весов.
    scheduler (torch.optim.lr_scheduler.LambdaLR): Планировщик для обновления скорости обучения.
    device (torch.device): Устройство для выполнения расчетов.

    Возвращает:
    tuple: Средние значения потерь и точности за один эпизод.
    """
    model.train()
    total_loss = 0
    total_acc = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=True)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Прямой проход через модель
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        logits = outputs.logits  # Логиты - выход модели
        loss = outputs.loss  # Потери модели
        acc = binary_accuracy(logits, labels)  # Вычисляем точность

        # Обратный проход и обновление параметров модели
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Обновляем прогресс-бар с текущими значениями потерь и точности
        progress_bar.set_postfix({'loss': loss.item(), 'accuracy': acc.item()})

        # Накопление потерь и точности для вычисления средних значений
        total_loss += loss.item()
        total_acc += acc.item()

    # Возвращаем средние значения потерь и точности
    return total_loss / len(train_loader), total_acc / len(train_loader)


def test_epoch(model, test_loader, device):
    """
    Тестирование модели в эпоху.

    Параметры:
    model (transformers.models.bert): Модель для оценки.
    test_loader (DataLoader): Загрузчик тестовых данных.
    device (torch.device): Устройство для выполнения расчетов.

    Возвращает:
    tuple: Средние значения потерь и точности на тестовом наборе данных.
    """
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Перемещение данных на устройство (GPU или CPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Прямой проход через модель (без вычисления градиентов)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits  # Логиты - выход модели
            loss = outputs.loss  # Потери модели
            acc = binary_accuracy(logits, labels)  # Вычисляем точность

            # Накопление потерь и точности
            total_loss += loss.item()
            total_acc += acc.item()

    return total_loss / len(test_loader), total_acc / len(test_loader)


def train_model(model, tokenizer, train_loader, test_loader, optimizer, scheduler, device, epochs=5, patience=2):
    """
    Обучение модели BertForSequenceClassification с применением ранней остановки и сохранением наилучшей модели.

    Параметры:
    model (transformers.models.bert): Модель для обучения.
    tokenizer (transformers.PreTrainedTokenizer): Токенизатор для обработки текста.
    train_loader (DataLoader): Загрузчик данных для обучения.
    test_loader (DataLoader): Загрузчик данных для тестирования.
    optimizer (torch.optim.Optimizer): Оптимизатор для обновления весов.
    scheduler (torch.optim.lr_scheduler.LambdaLR): Планировщик для обновления скорости обучения.
    device (torch.device): Устройство для выполнения расчетов.
    epochs (int): Количество эпох обучения.
    patience (int): Количество эпох без улучшения, после которых происходит ранняя остановка.

    Возвращает:
    torch.nn.Module: Обученная модель.
    dict: История обучения.
    """
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Обучаем модель
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")

        # Оценка модели на тестовом наборе данных
        test_loss, test_acc = test_epoch(model, test_loader, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print(f"Test loss: {train_loss:.4f}, Test accuracy: {train_acc:.4f}")

        # Early Stopping
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            epochs_without_improvement = 0
            save_model(model, tokenizer, "/model/")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered")
            break

    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }

    return model, history


def evaluate_model(model, test_loader, device, path):
    """
    Оценка модели на тестовом наборе данных с выводом отчета о классификации и матрицы ошибок.

    Параметры:
    model (transformers.models.bert): Модель для оценки.
    test_loader (DataLoader): Загрузчик тестовых данных.
    device (torch.device): Устройство для выполнения расчетов.

    Возвращает:
    str: Строка с отчетом о классификации.
    """
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)  # Получаем предсказания из логитов
            all_preds.extend(preds.cpu().numpy())  # Сохраняем предсказания
            all_labels.extend(labels.cpu().numpy())  # Сохраняем истинные метки

    # Возвращаем отчет о классификации
    print(classification_report(all_labels, all_preds, target_names=["non-suicide", "suicide"]))

    # Строим матрицу ошибок
    plot_confusion_matrix(all_labels, all_preds, path)
