import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.link_prediction import resource_allocation_index, jaccard_coefficient, adamic_adar_index, \
    preferential_attachment
from tqdm import tqdm
import json
import time
from random import shuffle
from pathlib import Path

weight = 'weight'


def reading_data(data_dir, data_split):
    ag = nx.Graph(
        nx.jit_graph(open(f'../DiffusionDataInput/{data_dir}/{data_split}/train/A.json', 'r', encoding='UTF-8').read()))
    print(nx.info(ag))
    tg = nx.Graph(
        nx.jit_graph(open(f'../DiffusionDataInput/{data_dir}/{data_split}/train/T.json', 'r', encoding='UTF-8').read()))
    print(nx.info(tg))
    atg = nx.Graph(nx.jit_graph(
        open(f'../DiffusionDataInput/{data_dir}/{data_split}/train/AT.json', 'r', encoding='UTF-8').read()))
    print(nx.info(atg))
    real_atg = nx.Graph(
        nx.jit_graph(open(f'../DiffusionDataInput/{data_dir}/{data_split}/test/AT.json', 'r', encoding='UTF-8').read()))
    print(nx.info(real_atg))
    return ag, tg, atg, real_atg


class DiffusionModel:
    def __init__(self, author_net, term_net, author_term_net):
        self.AG = author_net
        self.TG = term_net
        self.ATG = author_term_net
        self.atg_degree = {}
        self.ag_degree = {}
        self.tg_degree = {}
        for u in tqdm(author_term_net.nodes()):
            self.atg_degree[u] = self.ATG.degree(u, weight)
        for u in tqdm(author_net.nodes()):
            self.ag_degree[u] = self.AG.degree(u, weight)
        for u in tqdm(term_net.nodes()):
            self.tg_degree[u] = self.TG.degree(u, weight)

    def predict(self, u, v):

        # no openness, no self-weight
        predicted_score = 0

        # resources from authors
        predicted_score += sum(self.AG[u][w]['weight'] / self.ag_degree[u] * self.ATG[w][v][
            'weight'] / self.atg_degree[w] for w in self.AG.neighbors(u) if w in self.ATG.neighbors(v))

        # resources from terms
        predicted_score += sum(self.ATG[u][w]['weight'] / self.atg_degree[u] * self.TG[w][v][
            'weight'] / self.tg_degree[w] for w in self.TG.neighbors(v) if w in self.ATG.neighbors(u))
        return predicted_score

    def diffusion_prediction(self, ebunch):
        return ((u, v, self.predict(u, v)) for u, v in ebunch)


class CF:
    def __init__(self, a, b, sparse_matrix):
        self.X = sparse_matrix
        self.node_num = a
        self.num_node = b

    def user_predict(self, u, v):
        v_u = self.node_num[u]
        v_v = self.node_num[v]
        cf = zip(self.X[v_v].indices, self.X[v_v].data)
        return sum(cosine_similarity(self.X[v_u], self.X[i[0]])[0][0] * i[1] for i in cf)

    def item_predict(self, u, v):
        v_u = self.node_num[u]
        v_v = self.node_num[v]
        cf = zip(self.X[v_u].indices, self.X[v_u].data)
        return sum(cosine_similarity(self.X[v_v], self.X[i[0]])[0][0] * i[1] for i in cf)

    def user_cf_prediction(self, ebunch):
        return ((u, v, self.user_predict(u, v)) for u, v in ebunch)

    def item_cf_prediction(self, ebunch):
        return ((u, v, self.item_predict(u, v)) for u, v in ebunch)


class ContentBased:
    def __init__(self, a, b, c, s_atg, s_tg):
        self.node_num = a
        self.num_node = b
        self.term_num = c
        self.X = s_atg
        self.Y = s_tg

    def predict(self, u, v):
        u = self.node_num[u]
        cf = self.X[u].indices
        tt = [self.term_num[self.num_node[i]] for i in cf]
        v = self.term_num[v]
        value = np.mean([cosine_similarity(self.Y[v], self.Y[i])[0][0] for i in tt])
        return value

    def content_prediction(self, ebunch):
        return ((u, v, self.predict(u, v)) for u, v in ebunch)


class WeightedRA:
    def __init__(self, all_g):
        self.G = all_g
        self.weighted_degree = {u: all_g.degree(u, 'weight') for u in all_g.nodes}

    def predict(self, u, v):
        return sum((self.G[u][w]['weight'] + self.G[v][w]['weight']) / self.weighted_degree[w] for w in
                   nx.common_neighbors(self.G, u, v))

    def weighted_ra_prediction(self, ebunch):
        return ((u, v, self.predict(u, v)) for u, v in ebunch)


def run_experiment(data_dir, data_split, random_selection, n_fold):
    """
    :param data_dir: "ALL" or "dblp_baby"
    :param data_split: 2010, 2015 or 2018
    :param random_selection: if run on ALL dataset, sample edges for test, if run on dblp_baby, test all the possible edges
    :param n_fold: run n independent experiments
    :return:
    """
    print('-----reading-----')

    ag, tg, atg, validate_atg = reading_data(data_dir, data_split)

    node2id = {u: i for i, u in enumerate(atg.nodes)}
    id2node = {i: u for i, u in enumerate(atg.nodes)}
    term2id = {u: i for i, u in enumerate(tg.nodes)} 
    id2term = {i: u for i, u in enumerate(tg.nodes)}
    sparse_atg = nx.convert_matrix.to_scipy_sparse_matrix(atg)
    sparse_tg = nx.convert_matrix.to_scipy_sparse_matrix(tg)

    # randomly remove the edges
    print('composing graphs....')
    allg = nx.compose_all([ag, tg, atg])

    # random pick 1000 non-exist edges and existing edges
    print('sampling nodes....')
    common_authors = [a for a in ag.nodes() if a in validate_atg.nodes()]
    common_terms = [t for t in tg.nodes() if t in validate_atg.nodes()]
    print(f'{len(common_authors)} common authors, {len(common_terms)} common terms')

    print('selecting bunches....')
    testing_pairs_list = []
    if not random_selection:
        testing_pairs = [(a, t) for a in common_authors for t in common_terms if not atg.has_edge(a, t)]
        testing_pairs_list = [testing_pairs]
    else:
        for iter_ in range(n_fold):
            true_edges = []
            false_edges = []
            shuffle(common_authors)
            shuffle(common_terms)
            for a in common_authors:
                for t in common_terms:
                    if validate_atg.has_edge(a, t) and not atg.has_edge(a, t) and len(true_edges) < random_selection:
                        true_edges.append((a, t))
                    if not validate_atg.has_edge(a, t) and not atg.has_edge(a, t) and len(
                            false_edges) < random_selection:
                        false_edges.append((a, t))
                if len(true_edges) == random_selection and len(false_edges) == random_selection:
                    break
            testing_pairs_list.append(true_edges + false_edges)

    start = time.time()

    print("Initiating model...")
    cf_model = CF(node2id, id2node, sparse_atg)
    content_model = ContentBased(node2id, id2node, term2id, sparse_atg, sparse_tg)
    weighted_ra = WeightedRA(allg)
    d_model = DiffusionModel(ag, tg, atg)

    # prediction
    for n_fold, testing_pairs in enumerate(testing_pairs_list):
        methods = {
            'ra': resource_allocation_index(allg, testing_pairs),
            'jc': jaccard_coefficient(allg, testing_pairs),
            'aa': adamic_adar_index(allg, testing_pairs),
            'pa': preferential_attachment(allg, testing_pairs),
            'weighted_ra': weighted_ra.weighted_ra_prediction(ebunch=testing_pairs),
            'diffusion': d_model.diffusion_prediction(ebunch=testing_pairs),
            'content-based': content_model.content_prediction(ebunch=testing_pairs),
            'icf': cf_model.item_cf_prediction(ebunch=testing_pairs),
            # 'ucf': cf_model.user_cf_prediction(ebunch=testing_pairs),
            # 'semantic_diffusion':  d_model.diffusion_prediction(ebunch=testing_pairs),

        }

        print("Start prediction for {} pairs".format(len(testing_pairs)))

        if not random_selection:
            output_dir = f'../1st_revision_output/{data_dir}/output_{data_split}_timeline'
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = f'../1st_revision_output/{data_dir}/output_{data_split}_timeline_10fold_{n_fold + 10}'
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for key, value in methods.items():
            result = []
            count = 0
            print(key)
            for u, v, p in tqdm(value):
                count += 1
                result.append(
                    (u, v, p, bool(validate_atg.has_edge(u, v)),
                     validate_atg[u][v]['weight'] if validate_atg.has_edge(u, v) else 0)
                )
            result = sorted(result, key=lambda x: x[2], reverse=True)

            # record the results
            f = open(f'{output_dir}/{key}.json', 'w+', encoding='UTF-8')
            for line in result:
                f.write(json.dumps(line) + '\n')
            f.close()
        end = time.time()
        print(end - start)


if __name__ == '__main__':
    # Data read and preprocessing
    exp = 'dblp_baby'  # "dblp_baby" for small dataset and "ALL" for full dataset
    for data_split in [
        # 2010,
        2015,
        2018
    ]:
        print(data_split)
        run_experiment(exp, data_split, random_selection=None, n_fold=10)
