from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
print("Loading datasets for top words calculation...")
fake_df = pd.read_csv(DATA_DIR / "Fake.csv")
true_df = pd.read_csv(DATA_DIR / "True.csv")


def get_top_words(series: pd.Series, n: int = 20):
    clean_series = series.dropna().astype(str)
    cleaned_texts = clean_series.apply(clean_text)
    combined = " ".join(cleaned_texts)
    words = combined.split()
    counter = Counter(words)
    return counter.most_common(n)


print("Extracting top words for Fake News...")
fake_top = get_top_words(fake_df["text"])

print("Extracting top words for Real News...")
true_top = get_top_words(true_df["text"])


# -----------------------------
# Fake News Plot (Light Theme)
# -----------------------------
fake_words = [x[0] for x in reversed(fake_top)]
fake_counts = [x[1] for x in reversed(fake_top)]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 6), facecolor="#ffffff")
ax.set_facecolor("#f8fafc")

bars = ax.barh(fake_words, fake_counts, color="#ef4444", edgecolor="#cbd5e1", linewidth=1.0)
ax.set_title("Top 20 Words in Fake News", fontsize=16, fontweight="bold", pad=15, color="#0f172a")
ax.set_xlabel("Frequency", fontsize=12, fontweight="bold", color="#334155", labelpad=8)
ax.tick_params(axis="both", colors="#475569", labelsize=11)
ax.grid(axis="x", linestyle="--", alpha=0.5, color="#cbd5e1")
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_color("#cbd5e1")

plt.tight_layout()
fake_out = OUTPUTS_DIR / "fake_top_words.png"
plt.savefig(fake_out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved fake top words chart to {fake_out}")


# -----------------------------
# Real News Plot (Light Theme)
# -----------------------------
true_words = [x[0] for x in reversed(true_top)]
true_counts = [x[1] for x in reversed(true_top)]

fig, ax = plt.subplots(figsize=(10, 6), facecolor="#ffffff")
ax.set_facecolor("#f8fafc")

bars = ax.barh(true_words, true_counts, color="#10b981", edgecolor="#cbd5e1", linewidth=1.0)
ax.set_title("Top 20 Words in Real News", fontsize=16, fontweight="bold", pad=15, color="#0f172a")
ax.set_xlabel("Frequency", fontsize=12, fontweight="bold", color="#334155", labelpad=8)
ax.tick_params(axis="both", colors="#475569", labelsize=11)
ax.grid(axis="x", linestyle="--", alpha=0.5, color="#cbd5e1")
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_color("#cbd5e1")

plt.tight_layout()
real_out = OUTPUTS_DIR / "real_top_words.png"
plt.savefig(real_out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved real top words chart to {real_out}")