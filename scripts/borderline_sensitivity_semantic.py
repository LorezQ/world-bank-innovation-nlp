import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 20261005
N_SPLITS = 5

WEIGHTS = {
    "Low": 6600 / 95,
    "Moderate": 774 / 86,
    "High": 200 / 65,
}

# ------------------------------------------------------------
# Load embeddings and metadata
# ------------------------------------------------------------

X_all = np.load("artifacts/validation_embeddings.npy")
meta = pd.read_csv("artifacts/validation_embedding_metadata.csv")

meta["weight"] = meta["innovation_tier"].map(WEIGHTS)

definite_mask = meta["human_label"].isin(["Yes", "No"]).to_numpy()
borderline_mask = (meta["human_label"] == "Borderline").to_numpy()

X_definite = X_all[definite_mask]
X_borderline = X_all[borderline_mask]

definite = meta.loc[definite_mask].copy().reset_index(drop=True)
borderline = meta.loc[borderline_mask].copy().reset_index(drop=True)

definite["human_binary"] = (
    definite["human_label"] == "Yes"
).astype(int)

y = definite["human_binary"].to_numpy()
weights = definite["weight"].to_numpy()

# ------------------------------------------------------------
# Recreate the same five folds used in the primary evaluation
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED,
)

borderline_probabilities = []

for fold_number, (train_idx, _) in enumerate(
    cv.split(X_definite, y),
    start=1,
):
    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_SEED,
    )

    model.fit(
        X_definite[train_idx],
        y[train_idx],
        sample_weight=weights[train_idx],
    )

    probs = model.predict_proba(X_borderline)[:, 1]
    borderline_probabilities.append(probs)

# Average predictions from the five models
borderline["semantic_probability"] = np.mean(
    np.vstack(borderline_probabilities),
    axis=0,
)

borderline["semantic_prediction"] = (
    borderline["semantic_probability"] >= 0.50
).astype(int)

print("BORDERLINE SEMANTIC PREDICTIONS")
print("=" * 60)
print("Borderline cases:", len(borderline))
print(
    "Predicted positive:",
    int(borderline["semantic_prediction"].sum()),
)
print(
    "Predicted negative:",
    int((borderline["semantic_prediction"] == 0).sum()),
)

print("\nPredictions by original innovation tier:")
print(
    pd.crosstab(
        borderline["innovation_tier"],
        borderline["semantic_prediction"],
        margins=True,
    )
)

# ------------------------------------------------------------
# Combine with the existing OOF predictions for definite cases
# ------------------------------------------------------------

primary = pd.read_csv("artifacts/model_predictions.csv")

borderline_for_eval = borderline[
    [
        "sample_id",
        "innovation_tier",
        "high_innovation",
        "weight",
        "semantic_probability",
        "semantic_prediction",
    ]
].copy()

borderline_for_eval["keyword_prediction"] = (
    borderline_for_eval["high_innovation"].astype(int)
)

def weighted_metrics(data, pred_col):
    y_true = data["human_binary"].to_numpy()
    y_pred = data[pred_col].to_numpy()
    w = data["weight"].to_numpy()

    tp = w[(y_true == 1) & (y_pred == 1)].sum()
    fp = w[(y_true == 0) & (y_pred == 1)].sum()
    fn = w[(y_true == 1) & (y_pred == 0)].sum()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


print("\nSENSITIVITY ANALYSIS")
print("=" * 60)

for borderline_value, label in [(0, "No"), (1, "Yes")]:

    b = borderline_for_eval.copy()
    b["human_binary"] = borderline_value

    combined = pd.concat(
        [
            primary[
                [
                    "sample_id",
                    "human_binary",
                    "weight",
                    "keyword_prediction",
                    "semantic_prediction",
                ]
            ],
            b[
                [
                    "sample_id",
                    "human_binary",
                    "weight",
                    "keyword_prediction",
                    "semantic_prediction",
                ]
            ],
        ],
        ignore_index=True,
    )

    keyword = weighted_metrics(
        combined,
        "keyword_prediction",
    )

    semantic = weighted_metrics(
        combined,
        "semantic_prediction",
    )

    print(f"\nBorderline -> {label}")

    print(
        "Keyword:  "
        f"Precision={keyword[0]:.3f}  "
        f"Recall={keyword[1]:.3f}  "
        f"F1={keyword[2]:.3f}"
    )

    print(
        "Semantic: "
        f"Precision={semantic[0]:.3f}  "
        f"Recall={semantic[1]:.3f}  "
        f"F1={semantic[2]:.3f}"
    )

borderline.to_csv(
    "artifacts/borderline_semantic_predictions.csv",
    index=False,
)

print(
    "\nSaved Borderline predictions to "
    "artifacts/borderline_semantic_predictions.csv"
)
