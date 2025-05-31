from pyalex import Works, Authors
import pyalex
from pathlib import Path
import pandas as pd
import networkx as nx

def curate_graph_data(raw_data_dir=None, min_authors=200, max_authors=300, email=None):
    if raw_data_dir is None:
        root_dir = Path(__file__).resolve().parents[2]
        raw_data_dir = root_dir / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    if email is not None:
        pyalex.config.email = email

    works_iter_pages = Works().search_filter(
        title_and_abstract="computer science"
    ).filter(
        publication_year="2020|2021|2022|2023|2024",
        is_oa=True,
        language="en"
    ).paginate(per_page=200)

    print("Works obtained.")

    for page in works_iter_pages:
        works_iter = page
        break

    author_ids = set()
    works_buffer = []

    for work in works_iter:
        works_buffer.append(work)
        for author in work['authorships']:
            author_ids.add(author['author']['id'])
        if len(author_ids) >= min_authors:
            break

    author_ids = list(author_ids)[:max_authors]

    authors_data = []
    for author_id in author_ids:
        author = Authors().filter(id=author_id).get()[0]
        authors_data.append({
            'id': author_id,
            'name': author['display_name'],
            'institutions': [inst['institution']['id'] for inst in author['affiliations']],
            'publications_count': author['works_count'],
            'citation_count': author['cited_by_count']
        })
    authors_df = pd.DataFrame(authors_data)

    print("Authors obtained.")

    G = nx.Graph()
    G.add_nodes_from(author_ids)
    for work in works_buffer:
        ids = [author['author']['id'] for author in work['authorships'] if author['author']['id'] in author_ids]
        for u in ids:
            for v in ids:
                if u < v:
                    G.add_edge(u, v)

    print("Coauthorship edges obtained.")

    authors_df.to_csv(raw_data_dir / "authors.csv", index=False)
    edges = pd.DataFrame([
        {'source': u, 'target': v, 'weight': G[u][v].get('weight', 1)}
        for u, v in G.edges()
    ])
    edges.to_csv(raw_data_dir / "coauthorship_edges.csv", index=False)

    print(f"Saved authors and coauthorship edges to {raw_data_dir}")

if __name__ == "__main__":
    curate_graph_data()