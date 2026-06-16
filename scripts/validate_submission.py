import json
from pathlib import Path

import numpy as np

from main import run


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PUBLIC_QUERIES = ROOT / "data" / "public_queries.json"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    if path.stat().st_size < 500:
        raise ValueError(
            f"{path} is suspiciously small. "
            "This may be a Git LFS pointer instead of the real artifact."
        )


def load_public_queries():
    with PUBLIC_QUERIES.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["query"] for item in data]


def validate_artifacts() -> None:
    required = [
        ARTIFACTS / "index_vectors.npy",
        ARTIFACTS / "index_meta.json",
        ARTIFACTS / "bm25.npz",
    ]

    for path in required:
        require_file(path)

    vectors = np.load(ARTIFACTS / "index_vectors.npy", mmap_mode="r")
    if vectors.dtype != np.float32:
        raise ValueError(f"Expected float32 vectors, got {vectors.dtype}")
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vector matrix, got shape {vectors.shape}")
    if vectors.shape[1] != 384:
        raise ValueError(f"Expected embedding dimension 384, got {vectors.shape[1]}")

    print("Artifact validation passed.")
    print(f"Vector matrix shape: {vectors.shape}")


def validate_run_outputs() -> None:
    queries = load_public_queries()
    outputs = run(queries)

    if len(outputs) != len(queries):
        raise ValueError(f"Expected {len(queries)} result lists, got {len(outputs)}")

    for i, result in enumerate(outputs):
        if not isinstance(result, list):
            raise TypeError(f"Result {i} is not a list")
        if len(result) < 10:
            raise ValueError(f"Result {i} has fewer than 10 IDs")
        top10 = result[:10]
        if len(set(top10)) != len(top10):
            raise ValueError(f"Result {i} has duplicate IDs in top 10")
        if not all(isinstance(x, int) for x in top10):
            raise TypeError(f"Result {i} contains non-integer IDs")

    print("run(queries) output validation passed.")
    print(f"Validated {len(queries)} public queries.")


def main():
    validate_artifacts()
    validate_run_outputs()
    print("Submission validation completed successfully.")


if __name__ == "__main__":
    main()