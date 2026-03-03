import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load metrics
df = pd.read_csv("training_metrics_altitude.csv")

# Best epoch info
best_epoch = 99
best_row = df[df["Epoch"] == best_epoch].iloc[0]
best_val_mae = best_row["Val_MAE_km"]
best_val_acc = best_row["Val_Acc_%"]
best_train_mae = best_row["Train_MAE_km"]

fig, ax = plt.subplots(figsize=(12, 6))

# Plot lines
ax.plot(df["Epoch"], df["Train_MAE_km"], color="#4C72B0", linewidth=1.8, label="Train MAE (km)")
ax.plot(df["Epoch"], df["Val_MAE_km"],   color="#DD8452", linewidth=1.8, label="Val MAE (km)")

# Highlight best validation point (epoch 99)
ax.scatter(best_epoch, best_val_mae, color="#2E4057", s=120, zorder=5)
ax.scatter(best_epoch, best_train_mae, color="#4C72B0", s=80, zorder=5)

# Vertical dashed line at best epoch
ax.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

# Annotation box
annotation_text = (
    f"Epoch {best_epoch} — Best Val MAE\n"
    f"Val MAE = {best_val_mae:.2f} km\n"
    f"Val Accuracy = {best_val_acc:.2f}%"
)
ax.annotate(
    annotation_text,
    xy=(best_epoch, best_val_mae),
    xytext=(42.5, 9.2),
    fontsize=9.5,
    color="white",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#2E4057", edgecolor="#2E4057", linewidth=1.5),
    arrowprops=dict(arrowstyle="->", color=(0.5, 0.5, 0.5, 0.5), lw=1.5, connectionstyle="arc3,rad=-0.25",
                    linestyle="dashed"),
)

# Labels and title
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("MAE (km)", fontsize=12)
ax.set_title("Altitude Estimation — Train and Validation MAE per Epoch", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim(1, df["Epoch"].max())
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("mae_per_epoch.png", dpi=150)
plt.show()
print(f"Plot saved to mae_per_epoch.png")
