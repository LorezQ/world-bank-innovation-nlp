import pandas as pd
import numpy as np

# Load frozen primary predictions and bootstrap results
df = pd.read_csv("artifacts/model_predictions.csv")
boot = pd.read_csv("artifacts/bootstrap_results.csv")

def weighted_metrics(data, prediction):
    y = data["human_binary"].to_numpy()
    p = data[prediction].to_numpy()
    w = data["weight"].to_numpy()

    tp = w[(y == 1) & (p == 1)].sum()
    fp = w[(y == 0) & (p == 1)].sum()
    fn = w[(y == 1) & (p == 0)].sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

models = {
    "Keyword baseline": "keyword_prediction",
    "Semantic embeddings": "semantic_prediction",
}

rows = []

for model_name, pred_col in models.items():

    point = weighted_metrics(df, pred_col)

    prefix = (
        "keyword"
        if model_name == "Keyword baseline"
        else "semantic"
    )

    for metric in ["Precision", "Recall", "F1"]:

        col = f"{prefix}_{metric.lower()}"

        lower, upper = np.percentile(
            boot[col].dropna(),
            [2.5, 97.5],
        )

        rows.append({
            "Model": model_name,
            "Metric": metric,
            "Estimate": point[metric],
            "CI_lower_95": lower,
            "CI_upper_95": upper,
        })

results = pd.DataFrame(rows)

results.to_csv(
    "results/model_performance.csv",
    index=False,
)

print(results.round(3))
print("\nSaved to results/model_performance.csv")