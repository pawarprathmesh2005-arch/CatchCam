"""
CatchCam — Natural Language Search
=====================================
Type (or speak, once wired to the dashboard) a description like
"person in a red jacket" and get back the tracked people across all
cameras that best match, ranked by similarity.

This is what your voice assistant feature will call: transcribed speech
becomes the --query text, and this script does the actual matching.

Usage:
    python clip_search.py --embeddings clip_embeddings/cam1_clip_embeddings.json clip_embeddings/cam2_clip_embeddings.json --query "person in a red jacket" --top_k 5

Requires: torch, open_clip_torch
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import open_clip


def load_clip_text_encoder(device):
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)
    return model, tokenizer


def embed_text_query(query: str, model, tokenizer, device) -> np.ndarray:
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features[0].cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def search(embedding_paths, query: str, top_k: int = 5, out_dir: str = "search_results"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_clip_text_encoder(device)
    query_embedding = embed_text_query(query, model, tokenizer, device)

    results = []
    for path in embedding_paths:
        with open(path) as f:
            data = json.load(f)
        cam_id = data["camera_id"]
        for track_id, info in data["tracks"].items():
            track_embedding = np.array(info["embedding"])
            score = cosine_similarity(query_embedding, track_embedding)
            results.append({
                "camera_id": cam_id,
                "track_id": int(track_id),
                "similarity": round(score, 4),
                "first_seen_timestamp": info["first_seen_timestamp"],
                "last_seen_timestamp": info["last_seen_timestamp"],
            })

    results.sort(key=lambda r: -r["similarity"])
    top_results = results[:top_k]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    safe_query = "".join(c if c.isalnum() else "_" for c in query)[:40]
    out_file = out_path / f"search_{safe_query}.json"
    with open(out_file, "w") as f:
        json.dump({"query": query, "results": top_results}, f, indent=2)

    print(f"[CatchCam] Query: \"{query}\"")
    print(f"[CatchCam] Top {len(top_results)} matches:")
    for r in top_results:
        print(f"  {r['camera_id']} track {r['track_id']}  "
              f"(similarity: {r['similarity']}, "
              f"seen {r['first_seen_timestamp']:.2f}s-{r['last_seen_timestamp']:.2f}s)")
    print(f"\n[CatchCam] Saved -> {out_file}")

    return top_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", nargs="+", required=True,
                         help="Paths to CLIP embedding JSON files (one or more cameras)")
    parser.add_argument("--query", required=True, help="Natural language description to search for")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out_dir", default="search_results")
    args = parser.parse_args()

    search(
        embedding_paths=args.embeddings,
        query=args.query,
        top_k=args.top_k,
        out_dir=args.out_dir,
    )
