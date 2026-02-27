"""
Финальное обучение Node2Vec на полном графе (без train/test split)
и генерация top-N рекомендаций для каждого фильма.

Гиперпараметры взяты из лучшего trial #9 (v7, MRR=0.2131).

Использование:
    python learn.py
"""

import json
import os
import random
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pecanpy.pecanpy import SparseOTF
from tqdm import tqdm

from train import (
    SkipGramModel,
    build_skipgram_pairs,
    device,
    graph_to_edgelist,
    load_graph,
)


# ─── Гиперпараметры trial #9 (v7) ──────────────────────────────────────────────

BEST_PARAMS = {
    "dimensions": 256,
    "walk_length": 40,
    "window": 7,
    "num_walks": 5,
    "p": 1.476716415889729,
    "q": 0.4807451179092647,
    "lr": 0.005874753346631105,
    "num_neg": 11,
}

MAX_EPOCHS = 1
TOP_N = 100
BATCH_SIZE = 524288
OUTPUT_PATH = "data/output/recommendations_v1.json"
GRAPH_PATH = "data/input/graph_v1.json"
SEED = 42


# ─── Training ───────────────────────────────────────────────────────────────────


def train_final(graph, params, max_epochs):
    """Обучение на полном графе без early stopping, фиксированное число эпох."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".edgelist", delete=False) as f:
        edgelist_path = f.name

    try:
        graph_to_edgelist(graph, edgelist_path)
        pecanpy_graph = SparseOTF(p=params["p"], q=params["q"], workers=4)
        pecanpy_graph.read_edg(edgelist_path, weighted=False, directed=False)
        print("Generating walks...")
        walks = pecanpy_graph.simulate_walks(
            num_walks=params["num_walks"], walk_length=params["walk_length"]
        )
        print(f"Generated {len(walks)} walks")
    finally:
        os.unlink(edgelist_path)

    all_nodes = sorted(graph.keys())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    vocab_size = len(all_nodes)

    print("Building dataset...")
    pairs = build_skipgram_pairs(walks, node_to_idx, params["window"])
    print(f"Pairs: {len(pairs):,}")
    all_pairs = torch.tensor(pairs, dtype=torch.long, device=device)
    del pairs

    model = SkipGramModel(vocab_size, params["dimensions"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    n_pairs = all_pairs.size(0)
    cur_batch_size = BATCH_SIZE

    for epoch in range(1, max_epochs + 1):
        perm = torch.randperm(n_pairs, device=device)
        all_pairs = all_pairs[perm]

        total_loss = 0
        num_batches = 0
        n_batches = (n_pairs + cur_batch_size - 1) // cur_batch_size
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch}/{max_epochs}")
        for b in pbar:
            start = b * cur_batch_size
            end = min(start + cur_batch_size, n_pairs)
            batch = all_pairs[start:end]
            target = batch[:, 0]
            context = batch[:, 1]
            neg = torch.randint(0, vocab_size, (end - start, params["num_neg"]), device=device)

            try:
                loss = model(target, context, neg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                cur_batch_size = cur_batch_size // 2
                print(f"\n[OOM] Reducing batch_size to {cur_batch_size}")
                n_batches = (n_pairs + cur_batch_size - 1) // cur_batch_size
                continue

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

        if num_batches == 0:
            print(f"Epoch {epoch}: all batches OOM, skipping")
            continue
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")

    embeddings = model.target_embeddings.weight.detach().cpu().numpy()
    return all_nodes, node_to_idx, embeddings


# ─── Recommendations ────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_recommendations(all_nodes, embeddings, top_n):
    """Top-N рекомендаций для каждого фильма (без фильтрации соседей)."""
    emb_tensor = torch.tensor(embeddings, device=device, dtype=torch.float32)
    emb_norm = F.normalize(emb_tensor, dim=1)

    n = len(all_nodes)
    # top_n + 1 потому что одна позиция — это сам фильм (self), его выкинем
    k = min(top_n + 1, n)

    recommendations = {}
    batch_size = 4096

    for start in tqdm(range(0, n, batch_size), desc="Generating recommendations"):
        end = min(start + batch_size, n)
        batch_emb = emb_norm[start:end]
        sims = batch_emb @ emb_norm.T

        # Маскируем self
        for i in range(end - start):
            sims[i, start + i] = -2.0

        _, topk_indices = torch.topk(sims, k, dim=1)
        topk_indices = topk_indices.cpu().numpy()

        for i in range(end - start):
            node = all_nodes[start + i]
            recs = []
            for idx in topk_indices[i]:
                if idx != start + i:
                    recs.append(all_nodes[idx])
                if len(recs) == top_n:
                    break
            recommendations[node] = recs

    return recommendations


# ─── Main ────────────────────────────────────────────────────────────────────────


def main():
    print(f"Device: {device}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    graph = load_graph(GRAPH_PATH)

    print(f"\nTraining with best params (trial #9):")
    for k, v in BEST_PARAMS.items():
        print(f"  {k:<15} {v}")
    print(f"  {'max_epochs':<15} {MAX_EPOCHS}")
    print()

    all_nodes, _, embeddings = train_final(graph, BEST_PARAMS, MAX_EPOCHS)
    print(f"Embeddings shape: {embeddings.shape}")

    print(f"\nGenerating top-{TOP_N} recommendations...")
    recs = generate_recommendations(all_nodes, embeddings, TOP_N)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(recs)} recommendations to {OUTPUT_PATH}")

    try:
        subprocess.run(["git", "add", OUTPUT_PATH], check=True)
        subprocess.run(["git", "commit", "-m", f"recommendations v1: {len(recs)} movies, top-{TOP_N}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("Pushed to git")
    except subprocess.CalledProcessError as e:
        print(f"[git push failed: {e}]")


if __name__ == "__main__":
    main()
