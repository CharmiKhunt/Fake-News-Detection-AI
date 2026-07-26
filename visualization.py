from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
fake_df = pd.read_csv(DATA_DIR / "Fake.csv")
true_df = pd.read_csv(DATA_DIR / "True.csv")

# Labels
fake_df["label"] = 0
true_df["label"] = 1

# Merge
df = pd.concat([fake_df, true_df], ignore_index=True)

# Remove duplicates
df = df.drop_duplicates(subset=["title", "text"])

# Count labels
counts = df["label"].value_counts()

# Plot styling for Light Theme
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(7, 5), facecolor="#ffffff")
ax.set_facecolor("#f8fafc")

colors_list = ["#ef4444", "#10b981"]  # Red for Fake, Green for Real
categories = ["Fake News", "Real News"]
values = [counts.get(0, 0), counts.get(1, 0)]

bars = ax.bar(
    categories,
    values,
    color=colors_list,
    edgecolor="#cbd5e1",
    linewidth=1.2,
    width=0.45,
)

ax.set_title("Distribution of Fake and Real News", fontsize=16, fontweight="bold", pad=15, color="#0f172a")
ax.set_xlabel("News Category", fontsize=12, fontweight="bold", color="#334155", labelpad=8)
ax.set_ylabel("Number of Articles", fontsize=12, fontweight="bold", color="#334155", labelpad=8)

ax.tick_params(axis="both", colors="#475569", labelsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_color("#cbd5e1")

# Display the count above each bar
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 400,
        f"{int(height):,}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
    )

plt.tight_layout()
output_path = OUTPUTS_DIR / "class_distribution.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved class distribution visualization to {output_path}")