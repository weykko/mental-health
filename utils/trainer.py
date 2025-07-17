import torch
from tqdm import tqdm

from utils.utils import plot_training_results, binary_accuracy, save_model


def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_acc = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=True)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        logits = outputs.logits
        loss = outputs.loss  # Вычисляем потери
        acc = binary_accuracy(logits, labels)  # Вычисляем точность

        # Обратный проход
        loss.backward()

        # Клипаем градиенты
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Обновляем параметры
        optimizer.step()

        # Обновляем скорость обучения
        scheduler.step()

        # Обновляем прогресс-бар
        progress_bar.set_postfix({'loss': loss.item(), 'accuracy': acc.item()})

        total_loss += loss.item()
        total_acc += acc.item()

    # Возвращаем средние значения потерь и точности
    return total_loss / len(train_loader), total_acc / len(train_loader)


def evaluate_epoch(model, test_loader, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            loss = outputs.loss
            acc = binary_accuracy(logits, labels)

            total_loss += loss.item()
            total_acc += acc.item()

    return total_loss / len(test_loader), total_acc / len(test_loader)


def train_and_evaluate(model, tokenizer, train_loader, test_loader, optimizer, scheduler, device, epochs=5, patience=2):
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")

        print("\nEvaluating model...")
        test_loss, test_acc = evaluate_epoch(model, test_loader, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print(f"Test loss: {train_loss:.4f}, Test accuracy: {train_acc:.4f}")

        # Calculate metrics
        # accuracy = accuracy_score(actual_labels, predictions)
        # precision, recall, f1, _ = precision_recall_fscore_support(
        #     actual_labels, predictions, average='weighted'
        # )
        # test_loss = avg_train_loss  # Test loss is approximated by training loss in this case
        #
        # print(f"Test Accuracy: {accuracy:.4f}")
        # print(f"Test Precision: {precision:.4f}")
        # print(f"Test Recall: {recall:.4f}")
        # print(f"Test F1 Score: {f1:.4f}")

        # Early stopping check
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            epochs_without_improvement = 0
            save_model(model, tokenizer, "/model/")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }
    # Plotting the results
    plot_training_results(history, "plots/train_history.png")

    return model