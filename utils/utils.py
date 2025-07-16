import torch
import re
from tqdm import tqdm


def preprocess_text(text):
    if isinstance(text, str):
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
    else:
        return ""


def train_model(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=True)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters
        optimizer.step()

        # Update learning rate
        scheduler.step()

        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    return predictions, actual_labels


def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, device, epochs=3, patience=3):
    training_losses = []
    test_losses = []
    test_accuracies = []
    train_accuracies = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train the model
        avg_train_loss, train_accuracy = train_model(model, train_loader, optimizer, scheduler, device)
        training_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        print(f"Average training loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Evaluate on test set after each epoch
        print("\nEvaluating model...")
        predictions, actual_labels = evaluate_model(model, test_loader, device)

        # Calculate metrics
        accuracy = accuracy_score(actual_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_labels, predictions, average='weighted'
        )
        test_loss = avg_train_loss  # Test loss is approximated by training loss in this case

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")

        # Early stopping check
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

    # Plotting the results
    plot_training_results(training_losses, train_accuracies, test_losses, test_accuracies)

    return model

# Function to plot training and test results
def plot_training_results(train_losses, train_accuracies, test_losses, test_accuracies):
    epochs = len(train_losses)

    # Create subplots for loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Train and Test Loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='#97a6c4')
    ax1.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='#384860')
    ax1.legend(loc='upper left')

    # Create another y-axis for accuracy
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='#97a6c4')
    ax2.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', color='#384860')
    ax2.legend(loc='upper right')

    # Add a legend
    fig.tight_layout()
    # Show plot
    plt.title('Training and Test Metrics')
    plt.show()