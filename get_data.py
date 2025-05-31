import argparse
from pathlib import Path
from src.data.curate_graph_data import curate_graph_data
from src.data.build_graph import build_graph
from src.data.edge_split import split_graph_edges

def main():
    parser = argparse.ArgumentParser()

    root_dir = Path(__file__).resolve().parent
    raw_dir = root_dir / "data" / "raw"
    processed_dir = root_dir / "data" / "processed"

    parser.add_argument('--raw_dir', type=str, default=raw_dir)
    parser.add_argument('--processed_dir', type=str, default=processed_dir)
    parser.add_argument('--topic', type=str, default="computer science")
    parser.add_argument('--email', type=str, default="ramdassingh399+openalex@gmail.com")
    args = parser.parse_args()


    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    curate_graph_data(raw_data_dir=raw_dir, email=args.email, topic=args.topic)
    build_graph(raw_dir=raw_dir, processed_dir=processed_dir)
    split_graph_edges(processed_dir=processed_dir)

if __name__ == "__main__":
    main()