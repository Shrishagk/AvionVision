
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

history = pd.read_csv(
    ROOT / "models" / "efficientnet_history.csv"
)

# Accuracy plot
plt.figure(figsize=(9, 5))

plt.plot(history["epoch"], history["accuracy"], label="Training Accuracy")
plt.plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("EfficientNetB0 - Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(ROOT / "models" / "accuracy_curve.png")
plt.show()

# Loss plot
plt.figure(figsize=(9, 5))

plt.plot(history["epoch"], history["loss"], label="Training Loss")
plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNetB0 - Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(ROOT / "models" / "loss_curve.png")
plt.show()

print("Training curves saved!")
