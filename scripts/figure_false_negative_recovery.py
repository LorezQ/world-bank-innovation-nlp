import matplotlib.pyplot as plt

categories = [
    "Recovered by\nsemantic model",
    "Still missed",
]

counts = [10, 5]

fig, ax = plt.subplots(figsize=(6, 5))

bars = ax.bar(categories, counts)

ax.set_ylabel("Number of validation cases")
ax.set_title("Semantic recovery of keyword false negatives")
ax.set_ylim(0, 12)

labels = [
    "10\n(66.7%)",
    "5\n(33.3%)",
]

ax.bar_label(
    bars,
    labels=labels,
    padding=5,
    fontsize=11,
)

ax.grid(axis="y", alpha=0.20)

fig.tight_layout()

fig.savefig(
    "figures/false_negative_recovery.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    "figures/false_negative_recovery.pdf",
    bbox_inches="tight",
)

print("Saved:")
print("figures/false_negative_recovery.png")
print("figures/false_negative_recovery.pdf")