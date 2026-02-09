import torch
import torch.nn.functional as F

def binary_classification(d, n, epochs=10000, lr=0.001, store_weights=False):
    # Q5 Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Q1 Generate random matrix X(n,d)
    X = torch.randn(n, d, dtype=torch.float32, device=device)

    # Q2 Generate a label vector (n,1)
    Y = (X.sum(dim=1, keepdim=True) > 2).float().to(device)

    # Q3 Initialize weights for each W matrix with a specific sigma
    W1 = torch.randn(d, 48, device=device) * torch.sqrt(torch.tensor(2.0 / d))
    W2 = torch.randn(48, 16, device=device) * torch.sqrt(torch.tensor(2.0 / 48))
    W3 = torch.randn(16, 32, device=device) * torch.sqrt(torch.tensor(2.0 / 16))
    W4 = torch.randn(32, 1, device=device) * torch.sqrt(torch.tensor(2.0 / 32))

    # Enable gradient tracking
    W1.requires_grad_(True)
    W2.requires_grad_(True)
    W3.requires_grad_(True)
    W4.requires_grad_(True)

    if store_weights:
        # Store on CPU to save GPU memory
        W1_history = torch.zeros(epochs, W1.shape[0], W1.shape[1], device='cpu')
        W2_history = torch.zeros(epochs, W2.shape[0], W2.shape[1], device='cpu')
        W3_history = torch.zeros(epochs, W3.shape[0], W3.shape[1], device='cpu')
        W4_history = torch.zeros(epochs, W4.shape[0], W4.shape[1], device='cpu')

    loss_history = []

    for epoch in range(epochs):
        # Forward pass
        Z1 = X @ W1
        A1 = torch.relu(Z1)

        Z2 = A1 @ W2
        A2 = torch.relu(Z2)

        Z3 = A2 @ W3
        A3 = torch.relu(Z3)

        logits = A3 @ W4

        # Loss
        loss = F.binary_cross_entropy_with_logits(logits, Y)
        loss_history.append(loss.item())

        # Backward pass
        loss.backward()

        # Gradient descent update
        with torch.no_grad():
            W1 -= lr * W1.grad
            W2 -= lr * W2.grad
            W3 -= lr * W3.grad
            W4 -= lr * W4.grad

            if store_weights:
                W1_history[epoch] = W1.detach().cpu().clone()
                W2_history[epoch] = W2.detach().cpu().clone()
                W3_history[epoch] = W3.detach().cpu().clone()
                W4_history[epoch] = W4.detach().cpu().clone()

            W1.grad.zero_()
            W2.grad.zero_()
            W3.grad.zero_()
            W4.grad.zero_()

    if store_weights:
        weight_history = {
            'W1': W1_history,
            'W2': W2_history,
            'W3': W3_history,
            'W4': W4_history
        }
        return W1, W2, W3, W4, loss_history, weight_history

    return W1, W2, W3, W4, loss_history

