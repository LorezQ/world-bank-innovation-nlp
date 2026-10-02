import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

boot = pd.read_csv("artifacts/bootstrap_results.csv")

metrics = ["Precision", "Recall", "F1"]

# Primary point estimates from the frozen evaluation
keyword_point = np.array([0.745, 0.370, 0.495])
semantic_point = np.array([0.547, 0.643, 0.591])

# Bootstrap confidence intervals calculated from saved replications
keyword_cols = [
    "keyword_precision",
    "keyword_recall",
    "keyword_f1",
]

semantic_cols = [
    "semantic_precision",
    "semantic_recall",
    "semantic_f1",
]

keyword_ci = np.array([
    np.percentile(boot[col].dropna(), [2.5, 97.5])
    for col in keyword_cols
])

semantic_ci = np.array([
    np.percentile(boot[col].dropna(), [2.5, 97.5])
    for col in semantic_cols
])

# Convert confidence intervals into distances from point estimates
keyword_err = np.vstack([
    keyword_point - keyword_ci[:, 0],
    keyword_ci[:, 1] - keyword_point,
])

semantic_err = np.vstack([
    semantic_point - semantic_ci[:, 0],
    semantic_ci[:, 1] - semantic_point,
])

x = np.arange(len(metrics))
offset = 0.10

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(
    x - offset,
    keyword_point,
    yerr=keyword_err,
    fmt="o",
    capsize=5,
    label="Keyword baseline",
)

ax.errorbar(
    x + offset,
    semantic_point,
    yerr=semantic_err,
    fmt="o",
    capsize=5,
    label="Semantic embeddings",
)

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)

ax.set_ylabel("Design-weighted score")
ax.set_title("Keyword baseline vs. semantic embedding classifier")

ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    "figures/model_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    "figures/model_comparison.pdf",
    bbox_inches="tight",
)

print("Saved:")
print("figures/model_comparison.png")
print("figures/model_comparison.pdf")
