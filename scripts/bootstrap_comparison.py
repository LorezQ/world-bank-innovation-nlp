import numpy as np
import pandas as pd

RANDOM_SEED = 20261005
N_BOOTSTRAPS = 5000

df = pd.read_csv("artifacts/model_predictions.csv")

rng = np.random.default_rng(RANDOM_SEED)


def weighted_metrics(data, prediction_column):
    y = data["human_binary"].to_numpy()
    pred = data[prediction_column].to_numpy()
    w = data["weight"].to_numpy()

    tp = w[(y == 1) & (pred == 1)].sum()
    fp = w[(y == 0) & (pred == 1)].sum()
    fn = w[(y == 1) & (pred == 0)].sum()

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else np.nan
    )

    return precision, recall, f1


results = []

tiers = ["Low", "Moderate", "High"]

for _ in range(N_BOOTSTRAPS):
    pieces = []

    # Resample independently within each original sampling stratum.
    for tier in tiers:
        stratum = df[df["innovation_tier"] == tier]

        sampled_indices = rng.choice(
            stratum.index.to_numpy(),
            size=len(stratum),
            replace=True,
        )

        pieces.append(df.loc[sampled_indices])

    boot = pd.concat(pieces, ignore_index=True)

    keyword = weighted_metrics(boot, "keyword_prediction")
    semantic = weighted_metrics(boot, "semantic_prediction")

    results.append(
        {
            "keyword_precision": keyword[0],
            "keyword_recall": keyword[1],
            "keyword_f1": keyword[2],
            "semantic_precision": semantic[0],
            "semantic_recall": semantic[1],
            "semantic_f1": semantic[2],
            "diff_precision": semantic[0] - keyword[0],
            "diff_recall": semantic[1] - keyword[1],
            "diff_f1": semantic[2] - keyword[2],
        }
    )


boot = pd.DataFrame(results)
boot.to_csv("artifacts/bootstrap_results.csv", index=False)


def report(column):
    values = boot[column].dropna()

    lower = np.percentile(values, 2.5)
    median = np.percentile(values, 50)
    upper = np.percentile(values, 97.5)

    print(
        f"{column:20s} "
        f"median={median:.3f}  "
        f"95% CI=[{lower:.3f}, {upper:.3f}]"
    )


print("STRATIFIED BOOTSTRAP RESULTS")
print("=" * 68)

print("\nKeyword baseline")
for metric in [
    "keyword_precision",
    "keyword_recall",
    "keyword_f1",
]:
    report(metric)

print("\nSemantic model")
for metric in [
    "semantic_precision",
    "semantic_recall",
    "semantic_f1",
]:
    report(metric)

print("\nSemantic minus keyword")
for metric in [
    "diff_precision",
    "diff_recall",
    "diff_f1",
]:
    report(metric)

print(f"\nBootstrap replications: {N_BOOTSTRAPS}")
print("Saved to artifacts/bootstrap_results.csv")
