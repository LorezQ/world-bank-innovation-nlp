import pandas as pd

df = pd.read_csv("artifacts/model_predictions.csv")

print("MODEL ERROR ANALYSIS")
print("=" * 60)

# Raw observed errors (not weighted portfolio estimates)
for name, pred_col in [
    ("Keyword", "keyword_prediction"),
    ("Semantic", "semantic_prediction"),
]:
    tp = ((df["human_binary"] == 1) & (df[pred_col] == 1)).sum()
    fn = ((df["human_binary"] == 1) & (df[pred_col] == 0)).sum()
    fp = ((df["human_binary"] == 0) & (df[pred_col] == 1)).sum()
    tn = ((df["human_binary"] == 0) & (df[pred_col] == 0)).sum()

    print(f"\n{name}")
    print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")


# How predictions changed between methods
print("\n\nPREDICTION TRANSITIONS")
print("=" * 60)

transition = pd.crosstab(
    [df["human_label"], df["keyword_prediction"]],
    df["semantic_prediction"],
    margins=True,
)

print(transition)


# Focus specifically on keyword false negatives
keyword_fn = df[
    (df["human_binary"] == 1) &
    (df["keyword_prediction"] == 0)
].copy()

recovered = keyword_fn["semantic_prediction"].sum()

print("\n\nKEYWORD FALSE NEGATIVES")
print("=" * 60)
print("Keyword false negatives:", len(keyword_fn))
print("Recovered by semantic model:", recovered)
print(
    "Recovery rate:",
    f"{recovered / len(keyword_fn):.1%}"
    if len(keyword_fn) else "N/A"
)


# Focus on new false positives introduced by semantic model
new_fp = df[
    (df["human_binary"] == 0) &
    (df["keyword_prediction"] == 0) &
    (df["semantic_prediction"] == 1)
]

print("\nNew false positives introduced by semantic model:", len(new_fp))


# Save cases for later qualitative inspection
keyword_fn.to_csv(
    "artifacts/keyword_false_negatives.csv",
    index=False,
)

new_fp.to_csv(
    "artifacts/semantic_new_false_positives.csv",
    index=False,
)

print("\nSaved case lists to artifacts/")
