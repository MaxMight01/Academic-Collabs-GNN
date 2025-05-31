# Graph Neural Network for Academic Collaboration Predictions

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Predict future co-authorship links among researchers using a GNN-based link prediction pipeline, based on the problem statement given.

## 1. Runthrough Guide
### **Installing dependencies**
Simply enter
```sh
pip install -r requirements.txt
```

### **Fetching and preprocessing of data**
- Query [OpenAlex](https://docs.openalex.org/) for authors who have published a paper regarding computer science between 2020 and 2024.
- Builds `authors.csv` and `coauthorship_edges.csv` in `.\data\raw`, and converts to a PyG `Data` object(`graph_data.pt`) in `.\data\processed`. This is then split into train/validation/test sets.
- Arguments exist and the default ones are shown below.
    - `raw_dir` is where the `.csv` files go and `processed_dir` is where the `.pt` and train/val/test sets go.
    - The `topic` to be queried for may be changed.
    - OpenAlex uses your mail to enter the [polite pool](https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication#the-polite-pool) which has faster and more consistent response times.
```sh
python get_data.py \
  --raw_dir data/raw \
  --processed_dir data/processed \
  --topic "computer science" \
  --email ramdassingh399@gmail.com
```
#### Details:
**Data curation:** [`pyalex`](https://github.com/J535D165/pyalex) is used to query OpenAlex for papers in “computer science” from 2020–2024 (OA & English); up to 300 unique author IDs are extracted, with metadata (name, institution IDs, works_count, cited_by_count). An undirected coauthorship graph is made with authors as nodes and an edge forming between nodes if and only if the authors have done a collaboration.

**Graph construction**: The `.csv` files are read. Institutions are encoded as a one-hot, via `MultiLabelBinarizer`. Publications count and citations count remain as is, as integers. Produces a [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) `Data(x, edge_index)` object.

**Edge splitting**: Applies `torch_geometric.transforms.RandomLinkSplit(...)` on `graph_data.pt`, with a split of 10% validation and 10% testing. Saves the files as `train_data.pt`, `val_data.pt`, and `test_data.pt`.


### Training, evaluation, and visualisation
- Trains model and outputs per-epoch validation metrics (AUC/Accuracy/F1) to console.
- Arguments exist and the default ones are shown below.
    - `data_dir` is where the train/val/test sets are, `plot_dir` is where the plots are dumped to, and `log_dir` is where the metadata is logged onto.
    - `config.json` contains the recommended hyperparameters, although the individual ones may also be edited.

```sh
python train.py \
  --data_dir data/processed \
  --plot_dir data/plots \
  --log_dir data/log \
  --config src/config.json \
  --epochs 100 \
  --lr 0.01 \
  --hidden_dim 128 \
  --layers 2 \
  --dropout 0.3
```

#### Details:

**Model training:** Defines a GraphSAGE network ([`SAGEConv`](https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.conv.SAGEConv.html)), and defines a LinkPredictor (MLP) to score pairs of node embeddings. Uses `BCEWithLogitsLoss` + negative sampling on training edges. The model is trained on the training set. After each epoch, the evaluation metrics are computed with the validation set.

**Evaluation:** Computes note embeddings on the `edge_index` of the data, and scores both real links and negatives.

**Visualization:** Saves plots for PCA and t-SNE. Also output correlation between the evaluation score and publication count difference, evaluation score and citation count difference, and evaluation score and institution Jaccard index.

**Logging:** A JSON file is created consisting of the evaluated metrics of the model and the hyperparameters used.

### Raw coauthorship graph
- Simply visualises the coauthorship graph.
- Arguments exist and the default ones are shown below.
    - `data_dir` is where the `graph_data.pt` is picked up, and `graph.png` is dumped into `plot_dir`.
```sh
python graph.py \
  --data_dir data/processed \
  --plot_dir data/plots
```

#### Details:

Just a little bonus; visualizes the raw coauthorship graph, with node sizes scaled by the logarithm of citation counts.

## 2. Results & Insights

- Validation AUC/Accuracy/F1 reach ~ 0.95 / 0.90 / 0.90 by epoch 100.
- PCA + t-SNE plots don't show any clear clustering of authors by institution. However, authors with higher citation counts tend to group slighty more than usual in t-SNE plot.
- The raw graph plot highlights a lot of disjoint subgraphs.


## 3. Robustness & Next Steps
- Multiple seeds: run `train.py` with different `--epochs`, record logs.
- Hyperparameter sweeps: systematically vary the hyperparameters.
- Prediction: given a new author, predict if a future collaboration would occur between the author one of the nodes in the model.



Thank you for exploring! Feel free to file issues, suggest enhancements, or open pull requests.