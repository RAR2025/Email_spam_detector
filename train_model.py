import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


class ContextualBandit:
    def __init__(
        self,
        n_features: int,
        n_actions: int = 2,
        alpha: float = 0.5,
        epsilon: float = 0.3,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.97,
        l2: float = 1e-5,
        class_weight: tuple[float, float] = (1.0, 3.0),
        bias: float = 0.15,
    ) -> None:
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.l2 = l2
        self.class_weight = class_weight
        self.bias = bias
        rng = np.random.default_rng(42)
        self.weights = rng.normal(0.0, 0.01, size=(n_actions, n_features))
        self.reward_history: list[float] = []
        self.epsilon_history: list[float] = []
        self.correct_history: list[int] = []
        self.total_history: list[int] = []

    def predict(self, x: np.ndarray, greedy: bool = False) -> int:
        scores = self.weights @ x
        scores = scores.copy()
        scores[1] += self.bias
        if greedy or np.random.random() >= self.epsilon:
            return int(np.argmax(scores))
        return int(np.random.randint(self.n_actions))

    def update(self, x: np.ndarray, action: int, reward: int) -> None:
        weighted_reward = reward * self.class_weight[action]
        self.weights[action] += self.alpha * weighted_reward * x
        if self.l2 > 0:
            self.weights[action] -= self.l2 * self.weights[action]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        episodes: int = 25,
        shuffle: bool = True,
    ) -> None:
        n_samples = X.shape[0]
        for episode in range(episodes):
            indices = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(indices)
            episode_reward = 0
            correct = 0
            for idx in indices:
                x = X[idx]
                true_action = int(y[idx])
                action = self.predict(x)
                reward = 1 if action == true_action else -1
                self.update(x, action, reward)
                episode_reward += reward
                if action == true_action:
                    correct += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.reward_history.append(episode_reward / n_samples)
            self.epsilon_history.append(self.epsilon)
            self.correct_history.append(correct)
            self.total_history.append(n_samples)
            print(
                f"Episode {episode+1:02d}/{episodes} | "
                f"avg reward: {self.reward_history[-1]:+.4f} | "
                f"accuracy: {correct/n_samples:.4f} | "
                f"epsilon: {self.epsilon:.4f}"
            )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        preds = np.array([self.predict(x, greedy=True) for x in X])
        metrics = {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, preds).tolist(),
            "predictions": preds.tolist(),
        }
        return metrics

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.weights @ x for x in X])


def find_optimal_bias(score_train: np.ndarray, y_train: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    best_f1, best_bias = -1.0, 0.0
    for b in np.linspace(-6.0, 6.0, 1201):
        preds = (score_train[:, 1] + b >= score_train[:, 0]).astype(int)
        f1 = f1_score(y_train, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_bias = f1, float(b)
    return best_bias


def main() -> None:
    print("Loading preprocessed data ...")
    X_train = np.load(ARTIFACT_DIR / "X_train.npy")
    X_test = np.load(ARTIFACT_DIR / "X_test.npy")
    y_train = np.load(ARTIFACT_DIR / "y_train.npy")
    y_test = np.load(ARTIFACT_DIR / "y_test.npy")

    with open(ARTIFACT_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    hyperparams = {
        "alpha": 0.5,
        "epsilon": 0.3,
        "epsilon_min": 0.02,
        "epsilon_decay": 0.97,
        "episodes": 25,
        "l2": 1e-3,
        "class_weight": [1.0, 1.0],
        "bias": 0.0,
    }

    bandit = ContextualBandit(
        n_features=X_train.shape[1],
        n_actions=2,
        alpha=hyperparams["alpha"],
        epsilon=hyperparams["epsilon"],
        epsilon_min=hyperparams["epsilon_min"],
        epsilon_decay=hyperparams["epsilon_decay"],
        l2=hyperparams["l2"],
        class_weight=tuple(hyperparams["class_weight"]),
        bias=hyperparams["bias"],
    )

    print("Training Contextual Bandit ...")
    bandit.train(X_train, y_train, episodes=hyperparams["episodes"])

    train_scores = bandit.score_matrix(X_train)
    optimal_bias = find_optimal_bias(train_scores, y_train)
    bandit.bias = optimal_bias
    print(f"Optimal decision bias (from train): {optimal_bias:.4f}")

    print("Evaluating on test set ...")
    metrics = bandit.evaluate(X_test, y_test)
    print(
        f"Test -> acc: {metrics['accuracy']:.4f} | "
        f"prec: {metrics['precision']:.4f} | "
        f"rec: {metrics['recall']:.4f} | "
        f"f1: {metrics['f1']:.4f}"
    )

    np.savez(
        ARTIFACT_DIR / "model_weights.npz",
        weights=bandit.weights,
        reward_history=np.array(bandit.reward_history),
        epsilon_history=np.array(bandit.epsilon_history),
        correct_history=np.array(bandit.correct_history),
        total_history=np.array(bandit.total_history),
        bias=np.array([bandit.bias]),
    )

    metrics_to_save = {k: v for k, v in metrics.items() if k != "predictions"}
    hyperparams["bias"] = optimal_bias
    metrics_to_save["hyperparams"] = hyperparams
    metrics_to_save["feature_names"] = vectorizer.get_feature_names_out().tolist()
    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"Saved model + metrics to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
