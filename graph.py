import argparse
from pathlib import Path
from src.analysis.visualise import visualise_graph

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory containing train/val/test .pt files')
    parser.add_argument('--plot_dir', type=str, default='data/plots', help='Directory to save visualisations')
    args = parser.parse_args()

    processed_dir = Path(args.data_dir)
    plot_dir = Path(args.plot_dir)

    visualise_graph(graph_data_path=processed_dir / "graph_data.pt", save_path=plot_dir / "graph.png")

if __name__ == "__main__":
    main()