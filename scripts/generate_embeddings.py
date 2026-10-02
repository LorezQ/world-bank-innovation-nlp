from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
STEP_SIZE = CHUNK_SIZE - CHUNK_OVERLAP

LABELS_FILE = "data/validation_labels.csv"
SOURCE_FILE = "baseline-reproduction/data/processed/processed_dataset.xlsx"
Path("artifacts").mkdir(exist_ok=True)

OUTPUT_EMBEDDINGS = "artifacts/validation_embeddings.npy"
OUTPUT_METADATA = "artifacts/validation_embedding_metadata.csv"

print("Loading validation labels...")
labels = pd.read_csv(LABELS_FILE)

print("Loading reproduced World Bank project data...")
source = pd.read_excel(
    SOURCE_FILE,
    usecols=["project_id", "lessons"],
)

df = labels.merge(
    source,
    on="project_id",
    how="left",
    validate="one_to_one",
)

if df["lessons"].isna().any():
    missing = df.loc[df["lessons"].isna(), "project_id"].tolist()
    raise ValueError(
        f"Missing lesson text for {len(missing)} validation projects: {missing[:10]}"
    )

print(f"Loaded {len(df)} validation projects.")

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def make_chunks(text):
    """Split one lesson into overlapping token chunks."""
    token_ids = tokenizer.encode(
        str(text),
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )

    chunks = []

    for start in range(0, len(token_ids), STEP_SIZE):
        chunk_ids = token_ids[start:start + CHUNK_SIZE]

        if not chunk_ids:
            break

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
        )
        chunks.append(chunk_text)

        if start + CHUNK_SIZE >= len(token_ids):
            break

    return chunks


project_embeddings = []
chunk_counts = []

print(f"Generating embeddings for {len(df)} projects...")

for i, text in enumerate(df["lessons"], start=1):
    chunks = make_chunks(text)
    chunk_counts.append(len(chunks))

    # Embed every chunk independently.
    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Mean pooling creates one project-level semantic representation.
    project_embedding = np.mean(chunk_embeddings, axis=0)

    # Normalize the project-level vector.
    norm = np.linalg.norm(project_embedding)
    if norm > 0:
        project_embedding = project_embedding / norm

    project_embeddings.append(project_embedding)

    if i % 25 == 0 or i == len(df):
        print(f"Processed {i}/{len(df)} projects")


embeddings = np.vstack(project_embeddings)

np.save(OUTPUT_EMBEDDINGS, embeddings)

metadata = df[
    ["sample_id", "innovation_tier", "high_innovation", "human_label"]
].copy()

metadata["n_chunks"] = chunk_counts
metadata.to_csv(OUTPUT_METADATA, index=False)

print("\nFinished.")
print("Embedding matrix shape:", embeddings.shape)
print("Saved embeddings to:", OUTPUT_EMBEDDINGS)
print("Saved metadata to:", OUTPUT_METADATA)
