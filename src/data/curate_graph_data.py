from pyalex import Works, Authors
from pathlib import Path
import pandas as pd
import networkx as nx

root_dir = Path(__file__).resolve().parents[2]
raw_data_dir = root_dir / "data" / "raw"

works_iter_pages = Works().search_filter(
    title_and_abstract = "computer science"
).filter(
    publication_year = "2020|2021|2022|2023|2024",
    is_oa = True,
    language = "en"
).paginate(per_page = 200)

for page in works_iter_pages:
    works_iter = page
    break

author_ids = set()
works_buffer = []
for work in works_iter:
    works_buffer.append(work)
    for author in work['authorships']:
        author_ids.add(author['author']['id'])
    if len(author_ids) >= 200:
        break
author_ids = list(author_ids)[:300]

authors_data = []
for author_id in author_ids:
    author = Authors().filter(id = author_id).get()[0]
    authors_data.append({
        'id': author_id,
        'name': author['display_name'],
        'institutions': [inst['institution']['id'] for inst in author['affiliations']],
        'publications_count': author['works_count'],
        'citation_count': author['cited_by_count']
    })
authors_df = pd.DataFrame(authors_data)

G = nx.Graph()
G.add_nodes_from(author_ids)
for work in works_buffer:
    ids = [author['author']['id'] for author in work['authorships'] if author['author']['id'] in author_ids]
    for u in ids:
        for v in ids:
            if u < v:
                G.add_edge(u, v)

authors_df.to_csv(raw_data_dir / "authors.csv", index=False)
edges = pd.DataFrame([
    {'source': u, 'target': v, 'weight': G[u][v].get('weight', 1)}
    for u, v in G.edges()
])
edges.to_csv(raw_data_dir / "coauthorship_edges.csv", index=False)