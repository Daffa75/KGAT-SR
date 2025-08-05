import math
import time
import random
import torch
import numpy as np
import pandas as pd
import networkx as nx

from kg import KGraph
from torch.utils.data import Dataset, DataLoader
import pickle

########################################## Evaluation #########################################
def getHitRatio(ranklist, targetItem):
    for item in ranklist:
        if item == targetItem:
            return 1
    return 0


def getNDCG(ranklist, targetItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == targetItem:
            return math.log(2) / math.log(i + 2)
    return 0


def getMRR(ranklist, targetItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == targetItem:
            return 1.0 / (i + 1)
    return 0


def metrics(ranklist, targetItem):
    hr = getHitRatio(ranklist, targetItem)
    ndcg = getNDCG(ranklist, targetItem)
    mrr = getMRR(ranklist, targetItem)
    return hr, ndcg, mrr


######################################### Data Loader #########################################
class HistDataset(Dataset):
    def __init__(self, df, idx_list, attr_size, hist_max_len):
        self.data = df.values  # [user, item, timestamp]
        self.idx_list = idx_list
        self.MASK = 0
        self.attr_size = attr_size
        self.hist_max_len = hist_max_len

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        l, r, t = self.idx_list[item]
        submat = self.data[l:r]
        h_uid, h_items, h_attrs = submat.T
        uid, t_item, _ = self.data[t]
        assert np.all(h_uid == uid)

        h_attrs[-1] = [self.MASK] * self.attr_size
        h_attrs = np.array([i for i in h_attrs])
        n = len(h_items)
        if n < self.hist_max_len:
            h_items = np.pad(h_items, [0, self.hist_max_len - n], 'constant', constant_values=self.MASK)
            h_attrs = np.pad(h_attrs, [(0, self.hist_max_len - n), (0, 0)], 'constant', constant_values=self.MASK)
        return uid, np.array(list(h_items)).astype(np.long), h_attrs, t_item


class Loader():
    def __init__(self, args):
        self.dataset = args.dataset
        self.attr_size = args.attr_size
        self.neibor_size = args.neibor_size
        self.hist_min_len = args.hist_min_len
        self.hist_max_len = args.hist_max_len
        self.train_batch_size = args.batchSize
        self.eval_batch_size = args.batchSize
        self.n_workers = args.n_workers
        self.valid = args.valid
        self.n_neg = args.n_neg
        self.MASK = 0

        self.kg = KGraph(self.dataset, self.attr_size)
        self.node_neibors = self.kg.node_neibors
        self.n_entity, self.n_relation = self.kg.n_entity, self.kg.n_relation

        self.nodes_degree = self.kg.nodes_degree
        self.D_node = self.construct_D()
        self.adj_entity, self.adj_relation = self.construct_neibors_adj()
        self.all_items_list, self.train_dl, self.valid_dl = self.load_data()

    def construct_D(self):
        sorted_degree = sorted(self.nodes_degree.items(), key=lambda x: x[0])
        D_node = [i[0] for i in sorted_degree]
        return D_node

    def construct_neibors_adj(self):
        adj_entity = np.zeros([self.n_entity, self.neibor_size], dtype=np.int64)
        adj_relation = np.zeros([self.n_entity, self.neibor_size], dtype=np.int64)

        for node in range(self.n_entity):
            neighbors = self.node_neibors[node]
            n_neighbors = len(neighbors)
            # sample
            if n_neighbors >= self.neibor_size:
                sampled_indices = np.random.choice(neighbors, size=self.neibor_size, replace=False)
            else:
                sampled_indices = np.random.choice(neighbors, size=self.neibor_size, replace=True)

            adj_entity[node] = np.array([n for n in sampled_indices])
            adj_relation[node] = np.array([self.kg.G.get_edge_data(node, n)['rel'] for n in sampled_indices])
        return adj_entity, adj_relation

    def extract_subseq(self, n):
        idx_list = []
        for right in range(self.hist_min_len, n):
            left = max(0, right - self.hist_max_len)
            target = right
            idx_list.append([left, right, target])
        return np.asarray(idx_list)

    def get_idx(self, df):
        offset = 0
        train_idx_list = []
        valid_idx_list = []
        for n in df.groupby('user').size():
            train_idx_list.append(self.extract_subseq(n - 1) + offset)
            valid_idx_list.append(np.add([max(0, n - 1 - self.hist_max_len), n - 1, n - 1], offset))
            offset += n
        train_idx_list = np.concatenate(train_idx_list)
        valid_idx_list = np.stack(valid_idx_list)
        return train_idx_list, valid_idx_list

    def load_data(self):
        # The original function was attempting to load the data as a CSV, which is incorrect
        # for the current pickled tuple format (sequences, targets).
        # This revised function correctly computes the list of all unique items
        # and returns dummy values for the unused DataLoader objects.
    
        train_data = pickle.load(open('datasets/' + self.dataset + '/train.txt', 'rb'))
        train_sequences, train_targets = train_data
    
        test_data = pickle.load(open('datasets/' + self.dataset + '/test.txt', 'rb'))
        test_sequences, test_targets = test_data
    
        all_items = set()
        for seq in train_sequences:
            all_items.update(seq)
        all_items.update(train_targets)
    
        for seq in test_sequences:
            all_items.update(seq)
        all_items.update(test_targets)
    
        all_items_list = sorted(list(all_items))
    
        # The DataLoaders (train_dl, valid_dl) are not used by the main script,
        # which uses the `Data` class instead. Returning None for them.
        return all_items_list, None, None


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, kg, sub_graph=False, method='ggnn', sparse=False, shuffle=False):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.sub_graph = sub_graph
        self.sparse = sparse
        self.method = method
        self.kg = kg  # Store the knowledge graph object

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice(self, index):
        # Get batch data
        inputs_batch = self.inputs[index]
        targets_batch = self.targets[index]
        mask_batch = self.mask[index]

        # Initialize lists for graph construction
        items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
        for u_input in inputs_batch:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)

        # Construct session graphs for the batch
        for u_input in inputs_batch:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)

            A_in.append(u_A_in)
            A_out.append(u_A_out)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        
        # Combine adjacency matrices
        A = np.concatenate([A_in, A_out], axis=2)

        # Fetch knowledge graph attributes for the batch
        h_attrs_batch = []
        for u_input in inputs_batch:
            # Remove padding before fetching attributes
            u_input_no_padding = [i for i in u_input if i != 0]
            
            # Get attributes from the knowledge graph
            entity_attrs_path = self.kg.entity_seq_shortest_path(u_input_no_padding)
            attrs = [item[1] for item in entity_attrs_path]
            
            # Pad attributes to match the max sequence length
            padded_attrs = np.zeros((self.len_max, self.kg.sample_attr_size), dtype=np.int64)
            if len(attrs) > 0:
                 padded_attrs[:len(attrs), :] = attrs
            h_attrs_batch.append(padded_attrs)

        # The 'h_items' are the input sequences themselves
        h_items_batch = inputs_batch
        
        # The 't_item' are the targets
        t_item_batch = targets_batch

        # Return all 8 values as expected by the train_test function
        return alias_inputs, A, items, mask_batch, t_item_batch, h_items_batch, np.array(h_attrs_batch), t_item_batch