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

