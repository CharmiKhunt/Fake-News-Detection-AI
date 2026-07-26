from pathlib import Path
import json
import os

import joblib
import pandas as pd
import requests

from preprocessing import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LATEST_DIR = DATA_DIR / "latest"
MODELS_DIR = BASE_DIR / "models"
NEWSAPI_CACHE_FILE = LATEST_DIR / "newsapi_latest.csv"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
NEWSAPI_PAGE_SIZE = 100
NEWSAPI_CATEGORIES = [
    ("business", ["stock market", "share market", "business", "finance", "economy"]),
    ("sports", ["sports", "cricket", "football", "tennis", "nba"]),
    ("entertainment", ["celebrity", "bollywood", "fashion", "movie", "music"]),
    ("technology", ["technology", "tech", "ai", "startup", "gadgets"]),
    ("health", ["health", "wellness", "fitness", "medicine"]),
    ("science", ["science", "research", "space"]),
]


def load_legacy_dataset() -> pd.DataFrame:
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1
    return pd.concat([fake_df, true_df], ignore_index=True)


def load_latest_dataset() -> pd.DataFrame:
    if not LATEST_DIR.exists():
        return pd.DataFrame()

    csv_files = sorted(LATEST_DIR.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    latest_df = pd.concat(frames, ignore_index=True)

    if "label" not in latest_df.columns:
        raise ValueError(
            "Latest dataset must include a 'label' column with 0 for fake and 1 for real."
        )

    return latest_df


def fetch_newsapi_latest_articles() -> pd.DataFrame:
    if not NEWSAPI_KEY:
        return pd.DataFrame()

    rows = []
    base_url = "https://newsapi.org/v2/everything"

    for category, queries in NEWSAPI_CATEGORIES:
        for query in queries:
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": NEWSAPI_PAGE_SIZE,
                "apiKey": NEWSAPI_KEY,
            }

            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as exc:
                print(f"NewsAPI request failed for query '{query}': {exc}")
                continue

            for article in payload.get("articles", []):
                title = article.get("title") or ""
                content = article.get("content") or ""
                description = article.get("description") or ""
                text = " ".join(part for part in [title, description, content] if part).strip()

                if not text:
                    continue

                rows.append(
                    {
                        "title": title,
                        "text": text,
                        "source": (article.get("source") or {}).get("name", ""),
                        "published_at": article.get("publishedAt", ""),
                        "label": 1,
                        "category": category,
                        "url": article.get("url", ""),
                    }
                )

    if not rows:
        return pd.DataFrame()

    latest_news_df = pd.DataFrame(rows)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    latest_news_df.to_csv(NEWSAPI_CACHE_FILE, index=False)
    print(f"Saved NewsAPI cache to {NEWSAPI_CACHE_FILE}")
    return latest_news_df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "text" not in df.columns:
        if "content" in df.columns:
            df["text"] = df["content"]
        elif "article" in df.columns:
            df["text"] = df["article"]
        elif "title" in df.columns:
            df["text"] = df["title"]
        else:
            raise ValueError("Dataset must contain a 'text', 'content', 'article', or 'title' column.")

    if "title" in df.columns and "source" not in df.columns:
        df["text"] = df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    return df


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)

    legacy_df = load_legacy_dataset()
    latest_df = load_latest_dataset()
    newsapi_df = fetch_newsapi_latest_articles()

    if not latest_df.empty:
        print(f"Loaded latest dataset with {len(latest_df):,} rows.")
    if not newsapi_df.empty:
        print(f"Fetched {len(newsapi_df):,} latest NewsAPI articles.")

    df = pd.concat([legacy_df, latest_df, newsapi_df], ignore_index=True)
    df = normalize_columns(df)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    df = df[df["text"].str.strip().astype(bool)].copy()

    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    if "published_at" in df.columns and df["published_at"].notna().any():
        df = df.sort_values("published_at")

    X = df["clean_text"]
    y = df["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )

    param_distributions = {
        "tfidf__max_features": [10000, 15000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__max_df": [0.9, 0.95],
        "clf__C": [0.5, 1.0, 2.0],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    search.fit(X_train_text, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_text)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    vectorizer = best_model.named_steps["tfidf"]
    model = best_model.named_steps["clf"]

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(model, MODELS_DIR / "fake_news_model.pkl")

    metadata = {
        "accuracy": accuracy,
        "f1_score": f1,
        "best_params": search.best_params_,
        "train_rows": int(len(X_train_text)),
        "test_rows": int(len(X_test_text)),
        "total_rows": int(len(df)),
    }
    with open(MODELS_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Vectorizer saved successfully!")
    print("Model saved successfully!")
    print("Best params:")
    print(search.best_params_)
    print("\nAccuracy:")
    print(accuracy)
    print("\nF1 Score:")
    print(f1)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
