import json
import pickle
import re
import string
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "Dataset" / "spam.csv"

st.set_page_config(
    page_title="Spam Detector — RL Dashboard",
    page_icon="📧",
    layout="wide",
)


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


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df[["Category", "Message"]].dropna()
    df["label"] = df["Category"].map({"ham": 0, "spam": 1})
    df["msg_len"] = df["Message"].astype(str).apply(len)
    df["word_count"] = df["Message"].astype(str).apply(lambda s: len(s.split()))
    return df


@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(ARTIFACT_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    weights_data = np.load(ARTIFACT_DIR / "model_weights.npz")
    with open(ARTIFACT_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)
    return vectorizer, weights_data, metrics

def predict_message(message: str, vectorizer, weights: np.ndarray, bias: float = 0.0):
    cleaned = clean_text(message)
    if not cleaned.strip():
        return None
    x = vectorizer.transform([cleaned]).toarray()[0]
    scores = weights @ x
    scores = scores.copy()
    scores[1] += bias
    exp = np.exp(scores - scores.max())
    probs = exp / exp.sum()
    pred = int(np.argmax(scores))
    return {
        "cleaned": cleaned,
        "x": x,
        "scores": scores,
        "probs": probs,
        "pred": pred,
    }


def top_tokens_for_prediction(x: np.ndarray, feature_names, weights: np.ndarray, k: int = 10):
    spam_w = weights[1] * x
    ham_w = weights[0] * x
    active_idx = np.where(x > 0)[0]
    if len(active_idx) == 0:
        return [], []
    spam_pairs = sorted(
        [(feature_names[i], float(spam_w[i])) for i in active_idx],
        key=lambda p: p[1],
        reverse=True,
    )[:k]
    ham_pairs = sorted(
        [(feature_names[i], float(ham_w[i])) for i in active_idx],
        key=lambda p: p[1],
        reverse=True,
    )[:k]
    return spam_pairs, ham_pairs


df = load_dataset()
try:
    vectorizer, weights_data, metrics = load_artifacts()
    weights = weights_data["weights"]
    bias = float(weights_data["bias"][0]) if "bias" in weights_data.files else 0.0
    reward_history = weights_data["reward_history"]
    epsilon_history = weights_data["epsilon_history"]
    correct_history = weights_data["correct_history"]
    total_history = weights_data["total_history"]
    feature_names = metrics.get("feature_names", vectorizer.get_feature_names_out().tolist())
    artifacts_loaded = True
except Exception as e:
    st.error(f"Could not load model artifacts: {e}\n\nRun `python preprocess.py` and `python train_model.py` first.")
    artifacts_loaded = False

st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("**Contextual Bandit** (RL for classification)")
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Dataset")
st.sidebar.write(f"Total messages: **{len(df):,}**")
st.sidebar.write(f"Ham: **{(df['label']==0).sum():,}** | Spam: **{(df['label']==1).sum():,}**")
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Hyperparameters")
if artifacts_loaded:
    hp = metrics.get("hyperparams", {})
    st.sidebar.write(f"• Learning rate (α): `{hp.get('alpha','-')}`")
    st.sidebar.write(f"• ε start / min: `{hp.get('epsilon','-')}` / `{hp.get('epsilon_min','-')}`")
    st.sidebar.write(f"• ε decay: `{hp.get('epsilon_decay','-')}`")
    st.sidebar.write(f"• Episodes: `{hp.get('episodes','-')}`")
    st.sidebar.write(f"• L2: `{hp.get('l2','-')}`")
    st.sidebar.write(f"• Decision bias: `{hp.get('bias','-')}`")
st.sidebar.markdown("---")
st.sidebar.subheader("🔁 Retrain")
st.sidebar.code("python preprocess.py && python train_model.py", language="bash")
if artifacts_loaded and st.sidebar.button("🔄 Reload model", width="stretch"):
    st.cache_resource.clear()
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · NumPy")

st.title("📧 Email Spam Detector")
st.caption("A reinforcement-learning approach (Contextual Bandit) for spam classification")

if not artifacts_loaded:
    st.stop()

tab_predict, tab_data, tab_model = st.tabs(["🔮 Predict", "📊 Data Stats", "🤖 Model Stats"])


with tab_predict:
    st.header("Classify an Email")
    st.write("Paste any email/SMS content below and the bandit will predict **spam** or **ham**.")

    default_text = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 now."
    user_input = st.text_area("Email content", value=default_text, height=180)

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        predict_btn = st.button("🚀 Classify", type="primary", width="stretch")
    with col2:
        clear_btn = st.button("✖ Clear", width="stretch")

    if clear_btn:
        st.rerun()

    if predict_btn:
        result = predict_message(user_input, vectorizer, weights, bias=bias)
        if result is None:
            st.warning("Please enter some text to classify.")
        else:
            pred = result["pred"]
            label = "🚫 SPAM" if pred == 1 else "✅ HAM (not spam)"
            color = "#e74c3c" if pred == 1 else "#27ae60"
            st.markdown(
                f"<h2 style='color:{color}; margin-top: 0'>{label}</h2>",
                unsafe_allow_html=True,
            )
            probs = result["probs"]
            c1, c2 = st.columns(2)
            c1.metric("P(ham)", f"{probs[0]*100:.2f}%")
            c2.metric("P(spam)", f"{probs[1]*100:.2f}%")

            st.subheader("Confidence")
            st.progress(float(probs[pred]), text=f"{label} — {probs[pred]*100:.2f}%")

            spam_tokens, ham_tokens = top_tokens_for_prediction(
                result["x"], feature_names, weights, k=10
            )
            cs, ch = st.columns(2)
            with cs:
                st.markdown("**🔴 Top tokens pushing toward SPAM**")
                if spam_tokens:
                    tok_df = pd.DataFrame(spam_tokens, columns=["token", "score"])
                    st.dataframe(tok_df, hide_index=True, width="stretch")
                else:
                    st.write("_No active tokens_")
            with ch:
                st.markdown("**🟢 Top tokens pushing toward HAM**")
                if ham_tokens:
                    tok_df = pd.DataFrame(ham_tokens, columns=["token", "score"])
                    st.dataframe(tok_df, hide_index=True, width="stretch")
                else:
                    st.write("_No active tokens_")

            with st.expander("View preprocessed text"):
                st.code(result["cleaned"], language="text")


with tab_data:
    st.header("📊 Dataset Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total messages", f"{len(df):,}")
    c2.metric("Ham", f"{(df['label']==0).sum():,}")
    c3.metric("Spam", f"{(df['label']==1).sum():,}")
    c4.metric("Spam ratio", f"{(df['label']==1).mean()*100:.2f}%")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Class distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["Category"].value_counts()
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=["#27ae60", "#e74c3c"],
            startangle=90,
        )
        ax.axis("equal")
        st.pyplot(fig)

    with right:
        st.subheader("Message length by class")
        fig, ax = plt.subplots(figsize=(5, 4))
        for label, color in [("ham", "#27ae60"), ("spam", "#e74c3c")]:
            sub = df[df["Category"] == label]["msg_len"]
            ax.hist(sub, bins=40, alpha=0.6, label=label, color=color, edgecolor="black")
        ax.set_xlabel("Characters")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Word count distribution")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    df["word_count"].clip(upper=df["word_count"].quantile(0.99)).hist(
        bins=40, ax=ax, color="#3498db", edgecolor="black"
    )
    ax.set_xlabel("Words per message")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Top words in SPAM vs HAM")
    sw, hw = st.columns(2)
    for col, cat, color, target in [
        (sw, "spam", "#e74c3c", "🔴 Spam"),
        (hw, "ham", "#27ae60", "🟢 Ham"),
    ]:
        with col:
            st.markdown(f"**{target}**")
            tokens = []
            for msg in df[df["Category"] == cat]["Message"].astype(str):
                tokens.extend(
                    STEMMER.stem(t) for t in re.findall(r"[a-zA-Z]{3,}", msg.lower())
                    if t not in STOP_WORDS
                )
            common = Counter(tokens).most_common(20)
            fig, ax = plt.subplots(figsize=(5, 5))
            words = [w for w, _ in common][::-1]
            counts = [c for _, c in common][::-1]
            ax.barh(words, counts, color=color)
            ax.set_xlabel("Frequency")
            st.pyplot(fig)

    with st.expander("View raw sample messages"):
        sample_n = st.slider("Sample size", 5, 50, 10)
        st.dataframe(df[["Category", "Message"]].sample(sample_n, random_state=1), width="stretch")


with tab_model:
    st.header("🤖 Model Statistics")

    cm = np.array(metrics["confusion_matrix"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    c3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    c4.metric("F1 score", f"{metrics['f1']*100:.2f}%")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["ham", "spam"],
            yticklabels=["ham", "spam"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.subheader("Reward per Episode (training)")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(range(1, len(reward_history) + 1), reward_history, marker="o", color="#2980b9")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average reward")
        ax.set_title("Bandit learning curve")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.subheader("Epsilon decay (exploration → exploitation)")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(1, len(epsilon_history) + 1), epsilon_history, marker="o", color="#e67e22")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Training accuracy per episode")
    fig, ax = plt.subplots(figsize=(8, 3))
    accs = np.array(correct_history) / np.array(total_history)
    ax.plot(range(1, len(accs) + 1), accs, marker="o", color="#27ae60")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Train accuracy")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Top learned weights (per action)")
    col_a, col_b = st.columns(2)
    spam_w = weights[1]
    ham_w = weights[0]
    top_spam_idx = np.argsort(spam_w)[-15:][::-1]
    top_ham_idx = np.argsort(ham_w)[-15:][::-1]
    with col_a:
        st.markdown("**🔴 Heaviest SPAM weights**")
        st.dataframe(
            pd.DataFrame(
                [(feature_names[i], float(spam_w[i])) for i in top_spam_idx],
                columns=["token", "weight"],
            ),
            hide_index=True,
            width="stretch",
        )
    with col_b:
        st.markdown("**🟢 Heaviest HAM weights**")
        st.dataframe(
            pd.DataFrame(
                [(feature_names[i], float(ham_w[i])) for i in top_ham_idx],
                columns=["token", "weight"],
            ),
            hide_index=True,
            width="stretch",
        )
