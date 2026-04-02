# deepl/multiclass.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
import onnx

class SimpleNN(nn.Module):
    def __init__(self, in_features, num_classes=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x) 
        return x

class ClassTrainer:
    def __init__(self, X_train, y_train, model, eta=0.001, epochs=10, loss_fn=None, optimizer=None, device=None, class_weights=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        self.eta = eta
        self.epochs = epochs

        # Handle class weights, didnt help actually
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()

        #self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else optim.Adam(self.model.parameters(), lr=self.eta)
        self.loss_vector = torch.zeros(self.epochs)
        self.accuracy_vector = torch.zeros(self.epochs)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = self.loss_fn(outputs, self.y_train)
            loss.backward()
            self.optimizer.step()

            # Save metrics
            self.loss_vector[epoch] = loss.item()
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == self.y_train).float().mean()
            self.accuracy_vector[epoch] = acc.item()
            print(f'Epoch {epoch+1}/{self.epochs} - Loss: {loss.item():.4f}, Acc: {acc.item():.4f}')

    def test(self, X_test, y_test):
        self.model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y_test_tensor).float().mean().item()
            print(f'Test Accuracy: {acc:.4f}')
        return preds.cpu().numpy(), y_test_tensor.cpu().numpy()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = torch.argmax(outputs, dim=1)
        return preds.cpu().numpy()

    def save(self, filename="model.onnx"):
        dummy_input = torch.randn(1, self.X_train.shape[1]).to(self.device)
        torch.onnx.export(self.model, dummy_input, filename,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print(f'Model saved to {filename}')

    def evaluation(self, X_test=None, y_test=None):
        # Plot loss and accuracy during training
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(range(1, self.epochs+1), self.loss_vector.cpu().numpy(), marker='o')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1,2,2)
        plt.plot(range(1, self.epochs+1), self.accuracy_vector.cpu().numpy(), marker='o')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.show()

        if X_test is not None and y_test is not None:
            preds, y_true = self.test(X_test, y_test)

            # Confusion matrix
            cm = confusion_matrix(y_true, preds)
            print("Confusion Matrix:\n", cm)

            # Classification report
            print("Classification Report:\n", classification_report(y_true, preds, digits=4))

            # Individual metrics
            f1 = f1_score(y_true, preds, average='weighted')
            precision = precision_score(y_true, preds, average='weighted')
            recall = recall_score(y_true, preds, average='weighted')
            acc = accuracy_score(y_true, preds)

            print(f"Final Metrics on Test Set:\nAccuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# ImageNet CNN Components
# ─────────────────────────────────────────────────────────────────────────────

class ConvLayer(nn.Module):
    """
    Composite block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    Used for Blocks 1-5 in the ImageNetCNN architecture.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class ImageNetCNN(nn.Module):
    """
    CNN architecture for ImageNet classification.
    5 ConvLayer blocks followed by GlobalAvgPool and 2 FC layers.
    """
    def __init__(self, num_classes=1000, dropout=0.5):
        super(ImageNetCNN, self).__init__()

        self.features = nn.Sequential(
            ConvLayer(3,   64),   # Block 1: 224 -> 112
            ConvLayer(64,  128),  # Block 2: 112 -> 56
            ConvLayer(128, 256),  # Block 3: 56  -> 28
            ConvLayer(256, 512),  # Block 4: 28  -> 14
            ConvLayer(512, 512),  # Block 5: 14  -> 7
        )

        # Global Average Pooling: 7x7 -> 1x1, output is (batch, 512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes)
            # No Softmax here — CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)   # (batch, 512)
        x = self.classifier(x)
        return x


class CNNTrainer:
    """
    Trainer for ImageNetCNN using DataLoaders.
    Supports GPU selection, ONNX export, and plot saving.
    """
    def __init__(self, model, loss_fn, optimizer, scheduler=None, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def _run_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

            # Print every 10th batch
            if (batch_idx + 1) % 10 == 0:
                batch_acc = (preds == labels).float().mean().item()
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}  Batch Acc: {batch_acc:.4f}")

        avg_loss = total_loss / total
        avg_acc = correct / total
        return avg_loss, avg_acc

    def _validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += inputs.size(0)

        return correct / total

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")

            train_loss, train_acc = self._run_epoch(train_loader)
            val_acc = self._validate(val_loader)

            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)

            print(f"  >> Epoch {epoch} Summary — "
                  f"Loss: {train_loss:.4f}  "
                  f"Train Acc: {train_acc:.4f}  "
                  f"Val Acc: {val_acc:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()

    def save_onnx(self, filename="imagenet_cnn.onnx"):
        self.model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        torch.onnx.export(
            self.model, dummy_input, filename,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Model saved to {filename}")

    def save_plot(self, filename="training_plot.png"):
        epochs = range(1, len(self.train_loss_history) + 1)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_loss_history, marker='o')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_acc_history, marker='o', color='green')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.val_acc_history, marker='o', color='orange')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved to {filename}")
