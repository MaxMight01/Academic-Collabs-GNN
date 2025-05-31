import os
import math
import torch
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from networkx.algorithms.community import greedy_modularity_communities
from src.utils.utils import get_positive_edges, get_negative_edges

def visualise_graph(graph_data_path, save_path=None):
    data = torch.load(graph_data_path, weights_only=False)
    
    G = nx.Graph()
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)
    G.add_nodes_from(range(data.num_nodes))

    citation_counts = data.x[:, -1].tolist()
    node_sizes = [max(30, min(int(30 * math.log1p(c)), 200)) for c in citation_counts]

    pos = nx.spring_layout(G, seed=42, k=0.2)

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_nodes(G, pos, node_color="#1f77b4", node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.9)
    plt.title("Graph Visualization (Citation-Scaled Nodes)")
    plt.axis("off")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

class EmbeddingVisualizer:
    def __init__(self, embeddings, data, predictor=None):
        self.embeddings = embeddings.detach().cpu().numpy()
        self.device = embeddings.device
        self.data = data
        self.predictor = predictor

        self.features = data.x.detach().cpu().numpy()
        self.pub_counts = self.features[:, -2]
        self.citation_counts = self.features[:, -1]
        self.inst_matrix = self.features[:, :-2]

    def _save_or_show(self, path):
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def visualise_pca(self, path=None):
        z_pca = PCA(n_components=2).fit_transform(self.embeddings)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        feature_names = ['Publication Count', 'Citation Count']
        feature_data = [self.pub_counts, self.citation_counts]

        for i, ax in enumerate(axes):
            sc = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=feature_data[i], cmap='viridis', s=15)
            ax.set_title(f'PCA of Node Embeddings\nColored by {feature_names[i]}')
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            plt.colorbar(sc, ax=ax)

        plt.tight_layout()
        self._save_or_show(path)

    def visualise_tsne(self, path=None):
        tsne_result = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42).fit_transform(self.embeddings)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        scatter1 = axes[0].scatter(tsne_result[:, 0], tsne_result[:, 1], c=self.pub_counts, cmap='viridis', s=20)
        axes[0].set_title("t-SNE of Node Embeddings\nColored by Publication Count")
        axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")
        fig.colorbar(scatter1, ax=axes[0])

        scatter2 = axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=self.citation_counts, cmap='viridis', s=20)
        axes[1].set_title("t-SNE of Node Embeddings\nColored by Citation Count")
        axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")
        fig.colorbar(scatter2, ax=axes[1])

        plt.tight_layout()
        self._save_or_show(path)

    def visualise_tsne_institutions(self, path=None):
        tsne_result = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42).fit_transform(self.embeddings)

        num_inst = self.inst_matrix.sum(axis=1)

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=num_inst, cmap='plasma', s=20)
        ax.set_title("t-SNE of Embeddings\nColored by # of Institutions")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        plt.colorbar(sc, ax=ax, label="Affiliations count")
        plt.tight_layout()
        self._save_or_show(path)

    def analyze_feature_influence(self, sample_size=500):
        if self.predictor is None:
            raise ValueError("Predictor model is required for analyzing feature influence.")

        z = torch.tensor(self.embeddings).to(self.device)
        x = self.features

        pos = get_positive_edges(self.data)
        neg = get_negative_edges(self.data)

        def sample_edges(edges):
            edges = edges.cpu().numpy()
            if len(edges) > sample_size:
                indices = np.random.choice(len(edges), sample_size, replace=False)
                edges = edges[indices]
            return edges

        pos, neg = sample_edges(pos), sample_edges(neg)
        edges = np.vstack([pos, neg])

        pubs = x[:, -2]
        cites = x[:, -1]
        inst = x[:, :-2]

        scores, pub_sums, cit_sums, inst_sims = [], [], [], []
        with torch.no_grad():
            for u, v in edges:
                score = self.predictor(z[u].unsqueeze(0), z[v].unsqueeze(0)).sigmoid().item()
                scores.append(score)
                pub_sums.append(pubs[u] + pubs[v])
                cit_sums.append(cites[u] + cites[v])
                iu, iv = inst[u], inst[v]
                inter = np.minimum(iu, iv).sum()
                union = np.maximum(iu, iv).sum()
                inst_sims.append(inter / union if union > 0 else 0.0)

        def corr(a, b):
            return np.corrcoef(a, b)[0, 1]

        print("Correlation with predicted link score:")
        print(f"  publications sum : {corr(pub_sums, scores): .3f}")
        print(f"  citations sum    : {corr(cit_sums, scores): .3f}")
        print(f"  inst. similarity : {corr(inst_sims, scores): .3f}")