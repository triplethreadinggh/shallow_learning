import os
import matplotlib.pyplot as plt
from datetime import datetime
from shallow_learning import binary_classification
#from deepl.two_layer_binary_classification import binary_classification

def main():
    d = 10
    n = 500
    W1, W2, W3, W4, loss_history = binary_classification(d=d, n=n)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"crossentropyloss_{timestamp}.pdf"

    # Build full path
    full_path = os.path.join(output_dir, filename)

    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Binary Classification Loss")
    plt.savefig(full_path)
    print(f"Saved plot to {full_path}")

if __name__ == "__main__":
    main()

