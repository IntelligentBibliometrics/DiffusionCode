import networkx as nx
from tqdm import tqdm


def show_info(data_dir='dblp_baby', data_split=2015):
    ag = nx.Graph(
        nx.jit_graph(open(f'../DiffusionDataInput/{data_dir}/{data_split}/train/A.json', 'r', encoding='UTF-8').read()))
    print(nx.info(ag))
    tg = nx.Graph(
        nx.jit_graph(open(f'../DiffusionDataInput/{data_dir}/{data_split}/train/T.json', 'r', encoding='UTF-8').read()))
    print(nx.info(tg))
    atg = nx.Graph(
        nx.jit_graph(
            open(f'../DiffusionDataInput/{data_dir}/{data_split}/train/AT.json', 'r', encoding='UTF-8').read()))
    print(nx.info(atg))

    print('----------train-test---------')
    real_ag = nx.Graph(
        nx.jit_graph(open(f'../DiffusionDataInput/{data_dir}/{data_split}/test/A.json', 'r', encoding='UTF-8').read()))
    print(nx.info(real_ag))
    real_tg = nx.Graph(
        nx.jit_graph(open(f'../DiffusionDataInput/{data_dir}/{data_split}/test/T.json', 'r', encoding='UTF-8').read()))
    print(nx.info(real_tg))
    real_atg = nx.Graph(
        nx.jit_graph(open(f'../DiffusionDataInput/{data_dir}/{data_split}/test/AT.json', 'r', encoding='UTF-8').read()))
    print(nx.info(real_atg))

    common_authors = [a for a in ag.nodes() if a in real_ag.nodes()]
    common_terms = [t for t in tg.nodes() if t in real_tg.nodes()]
    print(f'{len(common_authors)} common authors, {len(common_terms)} common terms')

    print(f'{len(common_terms) * len(common_authors)} candidate edges.')

    existing_edges = 0
    appear_edges = 0
    negative_edges = 0
    for u in tqdm(common_authors):
        for v in common_terms:
            if atg.has_edge(u, v):
                existing_edges += 1
            elif real_atg.has_edge(u, v):
                appear_edges += 1
            else:
                negative_edges += 1
    print(f'{existing_edges} existing edges.')
    print(f'{appear_edges} test true edges.')
    print(f'{negative_edges} negative edges.')
    print(f'{negative_edges + appear_edges} candidate edges.')


show_info(data_dir='dblp_baby', data_split=2015)
