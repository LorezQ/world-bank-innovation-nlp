import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 20261005
N_SPLITS = 5

# Population and sample sizes from the validation design
WEIGHTS = {
    "Low": 6600 / 95,
    "Moderate": 774 / 86,
    "High": 200 / 65,
}

# Load embeddings and their corresponding metadata
X_all = np.load("artifacts/validation_embeddings.npy")
meta = pd.read_csv("artifacts/validation_embedding_metadata.csv")

if len(meta) != X_all.shape[0]:
    raise ValueError("Embedding rows and metadata rows do not match.")

# Primary evaluation: definite Yes/No cases only
mask = meta["human_label"].isin(["Yes", "No"]).to_numpy()

X = X_all[mask]
data = meta.loc[mask].copy().reset_index(drop=True)

data["human_binary"] = (data["human_label"] == "Yes").astype(int)
data["weight"] = data["innovation_tier"].map(WEIGHTS)

y = data["human_binary"].to_numpy()
weights = data["weight"].to_numpy()

# Same reproducible five-fold design defined in the protocol
cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED,
)

oof_probability = np.zeros(len(data))
oof_prediction = np.zeros(len(data), dtype=int)
fold_assignment = np.zeros(len(data), dtype=int)

for fold_number, (train_idx, test_idx) in enumerate(
    cv.split(X, y),
    start=1,
):
    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_SEED,
    )

    # Correct for the disproportionate validation sampling design
    model.fit(
        X[train_idx],
        y[train_idx],
        sample_weight=weights[train_idx],
    )

    probabilities = model.predict_proba(X[test_idx])[:, 1]

    oof_probability[test_idx] = probabilities
    oof_prediction[test_idx] = (probabilities >= 0.50).astype(int)
    fold_assignment[test_idx] = fold_number

data["fold"] = fold_assignment
data["semantic_probability"] = oof_probability
data["semantic_prediction"] = oof_prediction
data["keyword_prediction"] = data["high_innovation"].astype(int)


def weighted_metrics(y_true, y_pred, sample_weight):
    tp = sample_weight[(y_true == 1) & (y_pred == 1)].sum()
    fp = sample_weight[(y_true == 0) & (y_pred == 1)].sum()
    fn = sample_weight[(y_true == 1) & (y_pred == 0)].sum()
    tn = sample_weight[(y_true == 0) & (y_pred == 0)].sum()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


keyword = weighted_metrics(
    y,
    data["keyword_prediction"].to_numpy(),
    weights,
)

semantic = weighted_metrics(
    y,
    data["semantic_prediction"].to_numpy(),
    weights,
)

print("DESIGN-WEIGHTED OUT-OF-FOLD COMPARISON")
print("=" * 52)

print("\nKeyword baseline")
print(
    f"TN={keyword['TN']:.1f}  FP={keyword['FP']:.1f}  "
    f"FN={keyword['FN']:.1f}  TP={keyword['TP']:.1f}"
)
print(
    f"Precision={keyword['Precision']:.3f}  "
    f"Recall={keyword['Recall']:.3f}  "
    f"F1={keyword['F1']:.3f}"
)

print("\nSemantic embedding model")
print(
    f"TN={semantic['TN']:.1f}  FP={semantic['FP']:.1f}  "
    f"FN={semantic['FN']:.1f}  TP={semantic['TP']:.1f}"
)
print(
    f"Precision={semantic['Precision']:.3f}  "
    f"Recall={semantic['Recall']:.3f}  "
    f"F1={semantic['F1']:.3f}"
)

print("\nDifference: semantic minus keyword")
print(
    f"Precision={semantic['Precision'] - keyword['Precision']:+.3f}  "
    f"Recall={semantic['Recall'] - keyword['Recall']:+.3f}  "
    f"F1={semantic['F1'] - keyword['F1']:+.3f}"
)

data.to_csv(
    "artifacts/model_predictions.csv",
    index=False,
)

print("\nSaved out-of-fold predictions to:")
print("artifacts/model_predictions.csv")
