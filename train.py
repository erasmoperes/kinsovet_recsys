"""
Node2Vec на GPU + Optuna для подбора гиперпараметров.

Использование (Colab):
    from train import main
    main()

Параметры задаются константами в секции перед main().
"""

import json
import os
import random
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pecanpy.pecanpy import SparseOTF
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data ───────────────────────────────────────────────────────────────────────


def load_graph(path="graph.json"):
    with open(path) as f:
        raw = json.load(f)
    graph = {k: set(v) for k, v in raw.items()}

    num_nodes = len(graph)
    num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    max_edges = num_nodes * (num_nodes - 1) // 2
    density = num_edges / max_edges if max_edges > 0 else 0

    print(f"Нод: {num_nodes}")
    print(f"Рёбер: {num_edges}")
    print(f"Плотность: {density:.6f} ({density*100:.4f}%)")

    return graph


def split_edges(graph):
    """
    Убираем ровно 1 ребро из каждой ноды (где возможно).
    Нода помечается как обслуженная после удаления первого ребра.
    Ребро удаляется только если у обеих нод останется >= 1 ребро.
    Возвращает (train_graph, test_edges_dict), где test_edges_dict = {node: hidden_neighbor}.
    """
    all_edges = set()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            all_edges.add(tuple(sorted((node, neighbor))))

    all_edges = list(all_edges)
    random.shuffle(all_edges)

    degree = {node: len(neighbors) for node, neighbors in graph.items()}
    current_degree = dict(degree)

    served = set()
    test_edges_dict = {}
    removed_edges = set()

    for u, v in all_edges:
        u_needs = u not in served
        v_needs = v not in served
        if not u_needs and not v_needs:
            continue
        if current_degree[u] <= 1 or current_degree[v] <= 1:
            continue

        removed_edges.add((u, v))
        current_degree[u] -= 1
        current_degree[v] -= 1

        if u_needs:
            test_edges_dict[u] = v
            served.add(u)
        if v_needs:
            test_edges_dict[v] = u
            served.add(v)

    train_graph = defaultdict(set)
    for u, v in all_edges:
        if (u, v) not in removed_edges:
            train_graph[u].add(v)
            train_graph[v].add(u)

    not_served = len(graph) - len(served)
    print(f"Всего рёбер: {len(all_edges)}")
    print(f"Убрано рёбер: {len(removed_edges)}")
    print(f"Train рёбер: {len(all_edges) - len(removed_edges)}")
    print(f"Нод с тестовым ребром: {len(test_edges_dict)}")
    print(f"Нод без тестового ребра: {not_served} (степень 1, нельзя убрать)")

    return dict(train_graph), test_edges_dict


# ─── Model ──────────────────────────────────────────────────────────────────────


def build_skipgram_pairs(walks, node_to_idx, window):
    """Vectorized skip-gram pair generation using numpy."""
    walk_arrays = []
    for walk in walks:
        indices = [node_to_idx[n] for n in walk if n in node_to_idx]
        if len(indices) > 1:
            walk_arrays.append(np.array(indices, dtype=np.int64))

    all_targets = []
    all_contexts = []
    for arr in walk_arrays:
        n = len(arr)
        for offset in range(1, window + 1):
            # forward pairs
            if offset < n:
                all_targets.append(arr[:n - offset])
                all_contexts.append(arr[offset:])
            # backward pairs
            if offset < n:
                all_targets.append(arr[offset:])
                all_contexts.append(arr[:n - offset])

    targets = np.concatenate(all_targets)
    contexts = np.concatenate(all_contexts)
    return np.stack([targets, contexts], axis=1)


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.target_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, target_idx, context_idx, neg_idx):
        target_emb = self.target_embeddings(target_idx)
        context_emb = self.context_embeddings(context_idx)
        neg_emb = self.context_embeddings(neg_idx)

        pos_score = (target_emb * context_emb).sum(dim=1)
        pos_loss = -F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze(2)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)

        return (pos_loss + neg_loss).mean()


# ─── Training ───────────────────────────────────────────────────────────────────


def graph_to_edgelist(g, path):
    with open(path, "w") as f:
        written = set()
        for node, neighbors in g.items():
            for neighbor in neighbors:
                edge = tuple(sorted((node, neighbor)))
                if edge not in written:
                    f.write(f"{edge[0]}\t{edge[1]}\n")
                    written.add(edge)


def train_node2vec_gpu(
    g,
    dimensions=128,
    window=5,
    walk_length=20,
    num_walks=10,
    p=1.0,
    q=1.0,
    num_neg=5,
    batch_size=524288,
    lr=0.005,
    patience=3,
    min_delta=1e-4,
    workers=4,
):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".edgelist", delete=False) as f:
        edgelist_path = f.name

    try:
        graph_to_edgelist(g, edgelist_path)
        pecanpy_graph = SparseOTF(p=p, q=q, workers=workers)
        pecanpy_graph.read_edg(edgelist_path, weighted=False, directed=False)
        print("Generating walks...")
        walks = pecanpy_graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)
        print(f"Generated {len(walks)} walks")
    finally:
        os.unlink(edgelist_path)

    all_nodes = sorted(g.keys())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    vocab_size = len(all_nodes)

    print("Building dataset...")
    pairs = build_skipgram_pairs(walks, node_to_idx, window)
    print(f"Pairs: {len(pairs):,}")
    all_pairs = torch.tensor(pairs, dtype=torch.long, device=device)
    del pairs

    model = SkipGramModel(vocab_size, dimensions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_pairs = all_pairs.size(0)
    best_loss = float("inf")
    best_weights = model.target_embeddings.weight.detach().clone()
    wait = 0
    epoch = 0
    while True:
        epoch += 1
        perm = torch.randperm(n_pairs, device=device)
        all_pairs = all_pairs[perm]

        total_loss = 0
        num_batches = 0
        n_batches = (n_pairs + batch_size - 1) // batch_size
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch}")
        for b in pbar:
            start = b * batch_size
            end = min(start + batch_size, n_pairs)
            batch = all_pairs[start:end]
            target = batch[:, 0]
            context = batch[:, 1]
            neg = torch.randint(0, vocab_size, (end - start, num_neg), device=device)

            loss = model(target, context, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            best_weights = model.target_embeddings.weight.detach().clone()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs, best loss = {best_loss:.4f})")
                break

    embeddings = best_weights.cpu().numpy()
    return all_nodes, node_to_idx, embeddings


# ─── Evaluation ─────────────────────────────────────────────────────────────────


def evaluate_gpu(node_to_idx, embeddings, test_edges_dict, graph, train_graph, ks=(5, 10, 20)):
    """
    test_edges_dict: {node: hidden_neighbor} — ровно 1 скрытое ребро на ноду.
    Hit@K = нашли ли скрытого соседа в топ-K (0 или 1).
    train_graph используется для маскирования известных соседей при ранжировании.
    """
    original_degree = {node: len(neighbors) for node, neighbors in graph.items()}

    buckets = {
        "0-1": (0, 1),
        "1-2": (1, 2),
        "2-3": (2, 3),
        "3-6": (3, 6),
        "6-10": (6, 10),
        "10-15": (10, 15),
        "15-20": (15, 20),
        "20-25": (20, 25),
        "25-30": (25, 30),
        "30+": (30, float("inf")),
    }

    emb_tensor = torch.tensor(embeddings, device=device, dtype=torch.float32)
    emb_norm = F.normalize(emb_tensor, dim=1)

    results = {bucket: {f"hit@{k}": [] for k in ks} | {"mrr": []} for bucket in buckets}
    results["all"] = {f"hit@{k}": [] for k in ks} | {"mrr": []}

    max_k = max(ks)

    eval_nodes = [n for n in test_edges_dict if n in node_to_idx and test_edges_dict[n] in node_to_idx]
    eval_indices = torch.tensor([node_to_idx[n] for n in eval_nodes], device=device)
    true_indices = torch.tensor([node_to_idx[test_edges_dict[n]] for n in eval_nodes], device=device)

    batch_size = 512
    for start in tqdm(range(0, len(eval_nodes), batch_size), desc="Evaluating"):
        end = min(start + batch_size, len(eval_nodes))
        batch_nodes = eval_nodes[start:end]
        batch_idx = eval_indices[start:end]
        batch_true = true_indices[start:end].cpu().numpy()

        batch_emb = emb_norm[batch_idx]
        sims = batch_emb @ emb_norm.T

        # Маскируем себя
        sims[torch.arange(len(batch_idx), device=device), batch_idx] = -2.0

        # Маскируем известных соседей из train_graph
        for i, node in enumerate(batch_nodes):
            train_neighbors = train_graph.get(node, set())
            if train_neighbors:
                mask_indices = [node_to_idx[nb] for nb in train_neighbors if nb in node_to_idx]
                if mask_indices:
                    sims[i, torch.tensor(mask_indices, device=device, dtype=torch.long)] = -2.0

        topk_vals, topk_indices = torch.topk(sims, max_k, dim=1)
        topk_indices = topk_indices.cpu().numpy()

        ranked_all = torch.argsort(sims, dim=1, descending=True).cpu().numpy()

        for i, node in enumerate(batch_nodes):
            true_idx = batch_true[i]

            for k in ks:
                hit = 1.0 if true_idx in topk_indices[i, :k] else 0.0
                results["all"][f"hit@{k}"].append(hit)

            for rank_pos, idx in enumerate(ranked_all[i], 1):
                if idx == true_idx:
                    results["all"]["mrr"].append(1.0 / rank_pos)
                    break

            deg = original_degree.get(node, 0)
            for bucket_name, (lo, hi) in buckets.items():
                if lo <= deg < hi or (hi == float("inf") and deg >= lo):
                    for k in ks:
                        hit = 1.0 if true_idx in topk_indices[i, :k] else 0.0
                        results[bucket_name][f"hit@{k}"].append(hit)
                    for rank_pos, idx in enumerate(ranked_all[i], 1):
                        if idx == true_idx:
                            results[bucket_name]["mrr"].append(1.0 / rank_pos)
                            break
                    break

    # Печать
    print(f"\n{'Bucket':<10} {'Count':>6} ", end="")
    for k in ks:
        print(f"{'H@'+str(k):>8} ", end="")
    print(f"{'MRR':>8}")
    print("-" * (10 + 7 + 9 * len(ks) + 9))

    for bucket_name in ["all"] + list(buckets.keys()):
        data = results[bucket_name]
        count = len(data["mrr"])
        if count == 0:
            continue
        print(f"{bucket_name:<10} {count:>6} ", end="")
        for k in ks:
            val = np.mean(data[f"hit@{k}"]) if data[f"hit@{k}"] else 0
            print(f"{val:>8.4f} ", end="")
        mrr = np.mean(data["mrr"]) if data["mrr"] else 0
        print(f"{mrr:>8.4f}")

    return results


# ─── Optuna ─────────────────────────────────────────────────────────────────────


def objective(trial, train_graph, test_edges_dict, graph):
    ss = SEARCH_SPACE
    params = {
        "dimensions": trial.suggest_categorical("dimensions", ss["dimensions"]),
        "walk_length": trial.suggest_int("walk_length", **ss["walk_length"]),
        "window": trial.suggest_int("window", **ss["window"]),
        "num_walks": trial.suggest_int("num_walks", **ss["num_walks"]),
        "p": trial.suggest_float("p", **ss["p"]),
        "q": trial.suggest_float("q", **ss["q"]),
        "lr": trial.suggest_float("lr", **ss["lr"]),
        "num_neg": trial.suggest_int("num_neg", **ss["num_neg"]),
    }

    print(f"\n=== Trial {trial.number} ===")
    for k, v in params.items():
        print(f"  {k:<15} {v}")
    print()

    all_nodes, node_to_idx, embeddings = train_node2vec_gpu(
        train_graph,
        dimensions=params["dimensions"],
        window=params["window"],
        walk_length=params["walk_length"],
        num_walks=params["num_walks"],
        p=params["p"],
        q=params["q"],
        num_neg=params["num_neg"],
        lr=params["lr"],
    )

    results = evaluate_gpu(node_to_idx, embeddings, test_edges_dict, graph, train_graph)
    mrr = np.mean(results["all"]["mrr"])

    # Логируем доп. метрики
    for k in [5, 10, 20]:
        trial.set_user_attr(f"hit@{k}", np.mean(results["all"][f"hit@{k}"]))

    return float(mrr)


def _git_push(files, message):
    try:
        subprocess.run(["git", "add"] + files, check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[git push failed: {e}]")


def _save_callback(study, trial, study_name):
    csv_path = f"{study_name}.csv"
    db_path = f"{study_name}.db"
    study.trials_dataframe().to_csv(csv_path, index=False)
    best = study.best_trial
    print(f"[Trial {trial.number}] MRR={trial.value:.4f} | Best: #{best.number} MRR={best.value:.4f} | Saved {csv_path}")
    _git_push([csv_path, db_path], f"trial {trial.number}: MRR={trial.value:.4f}")


def run_optimization(train_graph, test_edges_dict, graph, n_trials=100, study_name="node2vec"):
    storage = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, train_graph, test_edges_dict, graph),
        n_trials=n_trials,
        callbacks=[lambda study, trial: _save_callback(study, trial, study_name)],
    )

    print("\n=== BEST TRIAL ===")
    best = study.best_trial
    print(f"MRR: {best.value:.4f}")
    print(f"Params: {best.params}")
    for k, v in best.user_attrs.items():
        print(f"  {k}: {v:.4f}")

    return study


# ─── Main ───────────────────────────────────────────────────────────────────────


STUDY_NAME = "node2vec_v3"
GRAPH_PATH = "graph.json"
SEED = 42
OPTIMIZE = True
N_TRIALS = 100

# Пространство поиска Optuna
SEARCH_SPACE = {
    "dimensions": [64, 128, 256],
    "walk_length": {"low": 10, "high": 40, "step": 5},
    "window": {"low": 2, "high": 5},
    "num_walks": {"low": 5, "high": 20, "step": 5},
    "p": {"low": 0.25, "high": 4.0, "log": True},
    "q": {"low": 0.25, "high": 4.0, "log": True},
    "lr": {"low": 1e-3, "high": 1e-2, "log": True},
    "num_neg": {"low": 3, "high": 10},
}

# Параметры для одиночного прогона (без Optuna)
DIMENSIONS = 128
WALK_LENGTH = 20
WINDOW = 5
NUM_WALKS = 10
P = 1.0
Q = 1.0
LR = 0.005
PATIENCE = 3
MIN_DELTA = 1e-4
BATCH_SIZE = 524288


def main():
    print(f"Device: {device}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    graph = load_graph(GRAPH_PATH)
    train_graph, test_edges_dict = split_edges(graph)

    if OPTIMIZE:
        run_optimization(train_graph, test_edges_dict, graph, n_trials=N_TRIALS, study_name=STUDY_NAME)
    else:
        all_nodes, node_to_idx, embeddings = train_node2vec_gpu(
            train_graph,
            dimensions=DIMENSIONS,
            window=WINDOW,
            walk_length=WALK_LENGTH,
            num_walks=NUM_WALKS,
            p=P,
            q=Q,
            lr=LR,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            batch_size=BATCH_SIZE,
        )
        print(f"Embeddings shape: {embeddings.shape}")
        evaluate_gpu(node_to_idx, embeddings, test_edges_dict, graph, train_graph)


if __name__ == "__main__":
    main()
