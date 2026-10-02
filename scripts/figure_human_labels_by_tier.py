import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/validation_labels.csv")

order = ["Low", "Moderate", "High"]
label_order = ["No", "Borderline", "Yes"]

counts = pd.crosstab(
    df["innovation_tier"],
    df["human_label"],
)

counts = counts.reindex(
    index=order,
    columns=label_order,
    fill_value=0,
)

percentages = counts.div(counts.sum(axis=1), axis=0) * 100

ax = percentages.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
)

ax.set_title("Human-adjudicated innovation labels by keyword tier")
ax.set_xlabel("Keyword-based innovation tier")
ax.set_ylabel("Share of validation sample (%)")

ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=0)

ax.legend(
    title="Human label",
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
)

# Add percentages inside sufficiently large segments
for container in ax.containers:
    labels = [
        f"{value:.0f}%" if value >= 8 else ""
        for value in container.datavalues
    ]

    ax.bar_label(
        container,
        labels=labels,
        label_type="center",
        fontsize=9,
    )

ax.grid(
    axis="y",
    alpha=0.20,
)

plt.tight_layout()

plt.savefig(
    "figures/human_labels_by_tier.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    "figures/human_labels_by_tier.pdf",
    bbox_inches="tight",
)

print("Counts:")
print(counts)

print("\nPercentages:")
print(percentages.round(1))

print("\nSaved:")
print("figures/human_labels_by_tier.png")
print("figures/human_labels_by_tier.pdf")
