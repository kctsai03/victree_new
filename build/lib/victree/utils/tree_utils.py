import itertools
from io import StringIO

import networkx as nx
import numpy as np
import torch
from typing import List, Tuple

from Bio import Phylo
from dendropy import Tree
from dendropy.calculate.treecompare import symmetric_difference
from networkx import is_arborescence

from utils import math_utils


def generate_fixed_tree(n_nodes: int, seed=0):
    return nx.random_tree(n=n_nodes, seed=seed, create_using=nx.DiGraph)


def get_unique_edges(T_list: List[nx.DiGraph], N_nodes: int = None) -> Tuple[List, torch.Tensor]:
    N_nodes = T_list[0].number_of_nodes() if N_nodes is None else N_nodes
    unique_edges_list = []
    unique_edges_count = torch.zeros(N_nodes, N_nodes, dtype=torch.int)
    for T in T_list:
        for uv in T.edges:
            if unique_edges_count[uv] == 0:
                unique_edges_count[uv] = 1
                unique_edges_list.append(uv)
            else:
                unique_edges_count[uv] += 1

    return unique_edges_list, unique_edges_count


def newick_from_eps_arr(eps_arr: np.ndarray):
    t = nx.DiGraph()
    for u, v in zip(*np.where(eps_arr > 0)):
        t.add_edge(u, v)
    return tree_to_newick(t)


def tree_to_newick(g: nx.DiGraph, root=None, weight=None):
    # make sure the graph is a tree
    assert is_arborescence(g)
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    # sorting makes sure same trees have same newick
    for child in sorted(g[root]):
        node_str: str
        if len(g[child]) > 0:
            node_str = tree_to_newick(g, root=child, weight=weight)
        else:
            node_str = str(child)

        if weight is not None:
            node_str += ':' + str(g.get_edge_data(root, child)[weight])
        subgs.append(node_str)
    return "(" + ','.join(subgs) + ")" + str(root)


def top_k_trees_from_sample(t_list, w_list, k: int = 0, by_weight=True, nx_graph=False):
    """
    Parameters
    ----------
    t_list list of sampled trees
    w_list sampled trees weights
    k number of unique top trees in output, if 0, return all unique trees
    by_weight bool, if true sorts by decreasing sum of weights.
        if false, sorts by decreasing number of trees (cardinality)
    nx_graph bool, if true nx.DiGraph object is returned, if false newick str is returned

    Returns
    -------
    list of tuples, top k trees depending on chosen order
    """
    unique_trees = {}
    for t, w in zip(t_list, w_list):
        t_newick: str = tree_to_newick(t)
        if t_newick not in unique_trees:
            unique_trees[t_newick] = {
                'nx-tree': t,
                'weight': 0.,
                'count': 0
            }
        diff = nx.difference(unique_trees[t_newick]['nx-tree'], t).size()
        assert diff == 0, \
            f"same string but different sets of edges: {t_newick} -> {[e for e in unique_trees[t_newick]['nx-tree'].edges]}," \
            f" {[e for e in t.edges]} | diff = {diff}"
        unique_trees[t_newick]['weight'] += w
        unique_trees[t_newick]['count'] += 1

        for alt_t in unique_trees:
            if alt_t != t_newick:
                # check effectively that all trees with different newick have
                # different sets of edges
                assert nx.difference(unique_trees[alt_t]['nx-tree'], t).size() > 0

    if k == 0:
        k = len(unique_trees)

    sorted_trees: [(str | nx.DiGraph, float)]
    sorted_trees = [(t_dat['nx-tree'] if nx_graph else t_str,
                     t_dat['weight'] if by_weight else t_dat['count'])
                    for t_str, t_dat in sorted(unique_trees.items(), key=lambda x: x[1]['weight'], reverse=True)]
    return sorted_trees[:k]


def networkx_tree_to_dendropy(T: nx.DiGraph, root) -> Tree:
    return Tree.get(data=tree_to_newick(T, root) + ";", schema="newick")


def calculate_SPR_distance(T_1: nx.DiGraph, T_2: nx.DiGraph):
    raise NotImplementedError("SPR distance not well defined for labeled trees.")
    T_1 = networkx_tree_to_dendropy(T_1, 0)
    T_2 = networkx_tree_to_dendropy(T_2, 0)
    spr_dist = None
    # spr_dist = MAF.get_spr_dist(T_1, T_2)
    return spr_dist


def calculate_Robinson_Foulds_distance(T_1: nx.DiGraph, T_2: nx.DiGraph):
    raise NotImplementedError("RF distance not well defined for labeled trees (only leaf labeled trees).")
    T_1 = networkx_tree_to_dendropy(T_1, 0)
    T_2 = networkx_tree_to_dendropy(T_2, 0)
    return symmetric_difference(T_1, T_2)


def calculate_Labeled_Robinson_Foulds_distance(T_1: nx.DiGraph, T_2: nx.DiGraph):
    "Package from: https://github.com/DessimozLab/pylabeledrf"
    T_1 = networkx_tree_to_dendropy(T_1, 0)
    # t1 = parseEnsemblLabels(T_1)
    T_2 = networkx_tree_to_dendropy(T_2, 0)
    # t2 = parseEnsemblLabels(T_2)
    # computeLRF(t1, t2)
    return symmetric_difference(T_1, T_2)


def calculate_graph_distance(T_1: nx.DiGraph, T_2: nx.DiGraph, roots=(0, 0), labeled_distance=True):
    if labeled_distance:
        # edge_match = lambda (u1,v1), (u2,v2) : u1==u2 and v1
        node_match = lambda u1, u2: u1 == u2
        distance = nx.graph_edit_distance(T_1, T_2, roots=roots, node_match=node_match)
    else:
        distance = nx.graph_edit_distance(T_1, T_2, roots=roots)
    return distance


def relabel_nodes(T, labeling):
    mapping = {o: i for i, o in enumerate(labeling)}
    return nx.relabel_nodes(T, mapping, copy=True)


def relabel_trees(T_list: list[nx.DiGraph], labeling):
    return [relabel_nodes(T, labeling) for T in T_list]


def distances_to_true_tree(true_tree, trees_to_compare: list[nx.DiGraph], labeling=None, labeled_distance=True):
    L = len(trees_to_compare)
    distances = np.zeros(L, )
    for l, T in enumerate(trees_to_compare):
        if labeling is not None:
            T = relabel_nodes(T, labeling)
        distances[l] = calculate_graph_distance(true_tree, T, labeled_distance=labeled_distance)
    return distances


def to_prufer_sequences(T_list: list[nx.DiGraph]):
    return [nx.to_prufer_sequence(T) for T in T_list]


def unique_trees(prufer_list: list[list[int]]):
    unique_seq = []
    unique_seq_idx = []
    for (i, seq) in enumerate(prufer_list):
        if seq in unique_seq:
            continue
        else:
            unique_seq.append(seq)
            unique_seq_idx.append(i)
    return unique_seq, unique_seq_idx


def unique_trees_and_multiplicity(T_list_or_prufer_seq_list):
    if type(T_list_or_prufer_seq_list[0]) == nx.DiGraph:
        undir_trees = to_undirected(T_list_or_prufer_seq_list)
        prufer_seq_list = to_prufer_sequences(undir_trees)
    else:
        prufer_seq_list = T_list_or_prufer_seq_list

    unique_seq = []
    unique_seq_idx = []
    multiplicity = []
    for (i, seq) in enumerate(prufer_seq_list):
        if seq in unique_seq:
            idx = unique_seq.index(seq)
            multiplicity[idx] += 1
        else:
            unique_seq.append(seq)
            unique_seq_idx.append(i)
            multiplicity.append(1)
    return unique_seq, unique_seq_idx, multiplicity


def to_undirected(T_list: list[nx.DiGraph]):
    return [nx.to_undirected(T) for T in T_list]


def get_two_step_connections(T: nx.DiGraph, u):
    two_step_neighbours = set()
    for v in nx.neighbors(T, u):
        two_step_neighbours.update(set(nx.neighbors(T, v)))

    return list(two_step_neighbours)


def get_all_two_step_connections(T: nx.DiGraph):
    node_two_step_neighbours_dict = {}
    for u in T.nodes:
        u_two_order_neighbours = get_two_step_connections(T, u)
        node_two_step_neighbours_dict[u] = u_two_order_neighbours

    return node_two_step_neighbours_dict


def remap_node_labels(T_list, perm):
    K = len(T_list[0].nodes)
    perm = list(range(K)) if perm is None else perm
    perm_dict = {i: perm[i] for i in range(K)}
    T_list_remapped = []
    for T in T_list:
        T_remapped = nx.relabel_nodes(T, perm_dict, copy=True)
        T_list_remapped.append(T_remapped)

    return T_list_remapped


def reorder_newick(nwk: str) -> str:
    nxtree = parse_newick(StringIO(nwk))
    return tree_to_newick(nxtree)


def get_all_tree_topologies(K):
    """
    Enumerate all labeled trees (rooted in 0) with their probability
    q(T) associated to the weighted graph which represents the current state
    of the variational distribution over the topology.
    Returns
    -------
    tuple with list of nx.DiGraph (trees) and tensor with normalized log-probabilities
    """
    c = 0
    tot_trees = math_utils.cayleys_formula(K)
    trees = []
    for pruf_seq in itertools.product(range(K), repeat=K - 2):
        unrooted_tree = nx.from_prufer_sequence(list(pruf_seq))
        rooted_tree = nx.dfs_tree(unrooted_tree, 0)

        trees.append(rooted_tree)
        c += 1

    assert tot_trees == c
    return trees


def star_tree(k: int) -> nx.DiGraph:
    t = nx.DiGraph()
    t.add_edges_from([(0, u) for u in range(1, k)])
    return t


def parse_newick(tree_file, config=None):
    """
    Parameters
    ----------
    tree_file: filepath. if newick string is desired, it's enough to
        pass StringIO(newick_string) instead
    config: config file for validation purposes on the number of nodes

    Returns
    -------
    nx.DiGraph tree
    """
    tree = Phylo.read(tree_file, 'newick')
    und_tree_nx = Phylo.to_networkx(tree)
    # Phylo names add unwanted information in unstructured way
    # find node numbers and relabel nx tree
    names_string = list(str(cl.confidence) if cl.name is None else cl.name for cl in und_tree_nx.nodes)
    mapping = dict(zip(und_tree_nx, map(int, names_string)))
    relabeled_tree = nx.relabel_nodes(und_tree_nx, mapping)
    tree_nx = nx.DiGraph()
    tree_nx.add_edges_from(relabeled_tree.edges())
    if config is not None:
        # validate config
        assert tree_nx.number_of_nodes() == config.n_nodes, "newick tree does not match the number of nodes K"
    return tree_nx
