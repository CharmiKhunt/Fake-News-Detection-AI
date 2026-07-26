from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
print("Loading datasets for word clouds...")
fake_df = pd.read_csv(DATA_DIR / "Fake.csv")
true_df = pd.read_csv(DATA_DIR / "True.csv")

# Apply preprocessing safely
print("Cleaning fake news text...")
fake_text_series = fake_df["text"].dropna().astype(str).apply(clean_text)
fake_text = " ".join(fake_text_series)

print("Cleaning real news text...")
true_text_series = true_df["text"].dropna().astype(str).apply(clean_text)
true_text = " ".join(true_text_series)

print("Generating Colorful Fake News WordCloud...")

# Fake News Word Cloud (Colorful)
fake_wc = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=150,
    random_state=42,
).generate(fake_text)

fig, ax = plt.subplots(figsize=(12, 6), facecolor="#ffffff")
ax.imshow(fake_wc, interpolation="bilinear")
ax.axis("off")
ax.set_title("Fake News Word Cloud", fontsize=18, fontweight="bold", pad=15, color="#0f172a")

plt.tight_layout()
fake_out = OUTPUTS_DIR / "fake_wordcloud.png"
plt.savefig(fake_out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Fake WordCloud completed and saved to {fake_out}")

# Real News Word Cloud (Colorful)
print("Generating Colorful Real News WordCloud...")
true_wc = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=150,
    random_state=42,
).generate(true_text)

fig, ax = plt.subplots(figsize=(12, 6), facecolor="#ffffff")
ax.imshow(true_wc, interpolation="bilinear")
ax.axis("off")
ax.set_title("Real News Word Cloud", fontsize=18, fontweight="bold", pad=15, color="#0f172a")

plt.tight_layout()
real_out = OUTPUTS_DIR / "real_wordcloud.png"
plt.savefig(real_out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Real WordCloud completed and saved to {real_out}")