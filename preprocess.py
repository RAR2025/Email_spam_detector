import re
import string
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import nltk
    nltk.data.find('corpora/stopwords')
except LookupError:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset" / "spam.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[$£€¥]", " currency ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [STEMMER.stem(tok) for tok in tokens if tok not in STOP_WORDS and len(tok) > 2]
    return " ".join(tokens)


def main() -> None:
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    if "Category" not in df.columns or "Message" not in df.columns:
        raise ValueError("Expected columns 'Category' and 'Message' in spam.csv")
    df = df[["Category", "Message"]].dropna()
    df["label"] = df["Category"].map({"ham": 0, "spam": 1})
    if df["label"].isna().any():
        raise ValueError("Found labels other than 'ham'/'spam'")
    print(f"Total samples: {len(df)} | ham: {(df['label']==0).sum()} | spam: {(df['label']==1).sum()}")

    print("Cleaning text ...")
    df["clean"] = df["Message"].apply(clean_text)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean"].values,
        df["label"].values,
        test_size=0.2,
        random_state=42,
        stratify=df["label"].values,
    )

    print("Fitting TF-IDF (unigrams + bigrams) ...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    np.save(ARTIFACT_DIR / "X_train.npy", X_train.toarray().astype(np.float32))
    np.save(ARTIFACT_DIR / "X_test.npy", X_test.toarray().astype(np.float32))
    np.save(ARTIFACT_DIR / "y_train.npy", y_train.astype(np.int64))
    np.save(ARTIFACT_DIR / "y_test.npy", y_test.astype(np.int64))

    with open(ARTIFACT_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    pd.DataFrame({
        "raw_message": X_train_text,
        "clean_message": X_train_text,
        "label": y_train,
    }).head(0).to_csv(ARTIFACT_DIR / "schema.csv", index=False)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Artifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
