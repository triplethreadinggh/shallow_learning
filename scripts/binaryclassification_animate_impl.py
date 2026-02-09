import os
import matplotlib.pyplot as plt
from datetime import datetime
from shallow_learning import binary_classification
from shallow_learning.animation import animate_weight_heatmap, animate_large_heatmap

def main():
    d = 200
    n = 40000
    epochs = 5000
    lr = 0.01
    dt = 0.04
    W1, W2, W3, W4, loss_history, weight_history= binary_classification(d=d, n=n, epochs=epochs, lr=lr, store_weights=True)

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
    plt.close()
    print(f"Saved plot to {full_path}")

    print("\nGenerating weight animations...")
    max_dim = max(
        weight_history['W1'].shape[1],
        weight_history['W1'].shape[2],
        weight_history['W2'].shape[1],
        weight_history['W2'].shape[2],
        weight_history['W3'].shape[1],
        weight_history['W3'].shape[2],
        weight_history['W4'].shape[1],
        weight_history['W4'].shape[2]
    )

    if max_dim > 500:
        print("Using large weight animation (matrix size > 500)")
        animate_func = animate_large_heatmap
    else:
        print("Using standard weight animation")
        animate_func = animate_weight_heatmap

    animate_func = animate_large_heatmap

    # Generate animation for each weight matrix W1, W2, W3, W4
    weight_names = ['W1', 'W2', 'W3', 'W4']

    for weight_name in weight_names:
        print(f"\nAnimating {weight_name}...")
        print(f"  Shape: {weight_history[weight_name].shape}")

        weight_data = weight_history[weight_name]

        # Call animation function
        animate_func(
            weight_data,
            dt=dt,
            file_name=f"{weight_name}_evolution_{timestamp}",  # unique filename
            title_str=f"{weight_name} Weight Evolution Over {epochs} Epochs"
        )

        print(f"  ✓ {weight_name} animation complete!")

    print("\n" + "="*60)
    print("ALL ANIMATIONS COMPLETE!")
    print("="*60)
    print(f"Loss plot saved to: {full_path}")
    print(f"Animation videos (.mp4) saved to: media/ folder")
    print("\nGenerated files:")
    for weight_name in weight_names:
        print(f"  - media/{weight_name}_evolution_{timestamp}.mp4")

if __name__ == "__main__":
    main()

