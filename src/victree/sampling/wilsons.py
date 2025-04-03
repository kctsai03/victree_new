import copy
import logging
import random
import pandas as pd

import networkx as nx
import numpy as np
import torch

import matplotlib
from networkx import maximum_spanning_arborescence

import matplotlib.pyplot as plt

from networkx.drawing.nx_pydot import graphviz_layout
from typing import Tuple

import time
from sampling.wilsons_helper import AddSelfLoops, RandomSuccessor, RandomPredecessor, Chance

'''
WILSON'S ALGORITHM:
  WilsonTreeWithRoot() builds spanning trees from a graph with a specified root.
  WilsonTree() builds a random spanning tree from a graph.
  Translated from source: https://dl.acm.org/doi/10.1145/237814.237880
'''

'''
INPUT: A vertex root r and graph G
OUTPUT: A random spanning tree named tree rooted at r
        where node i points to tree[i]
'''
def WilsonTreeWithRoot(G, r):
    n = len(G)

    Next = [None] * n
    InTree = [False] * n
    InTree[r] = True

    for i in range(n):
        u = i
        while not InTree[u]:
            Next[u] = RandomPredecessor(G, u)
            u = Next[u]
        u = i
        while not InTree[u]:
            InTree[u] = True
            u = Next[u]
    edges = tuple(sorted((b, a) for a, b in enumerate(Next) if b is not None))
    wilson_tree = nx.DiGraph()
    for u, v in edges:
        weight = (G[u][v]['weight'])
        wilson_tree.add_edge(u, v, weight=torch.log(weight))
    log_g = torch.tensor(0.0)
    for u, v in edges:
        log_g += calculate_log_g(u, v, G)
    return wilson_tree, edges, log_g

'''
INPUT: A graph G
OUTPUT: A random spanning tree, the edges, the total calls, and log_g
'''
def WilsonTree(G):
    n = len(G.nodes)
    G_no_self_loops = copy.deepcopy(G)

    def Attempt(epsilon):
        rand_predecessor_calls = 0
        Next = [None] * n
        InTree = [False] * n
        num_roots = 0

        for i in range(n):
            u = i
            while not InTree[u]:
                if Chance(epsilon):
                    Next[u] = None
                    InTree[u] = True
                    num_roots += 1
                    if num_roots > 1:
                        return None, rand_predecessor_calls
                else:
                    rand_predecessor_calls += 1
                    Next[u] = RandomPredecessor(G, u)
                    u = Next[u]
            u = i
            while not InTree[u]:
                InTree[u] = True
                u = Next[u]

        return Next, rand_predecessor_calls

    G = AddSelfLoops(G)
    epsilon = 1.0
    tree = None

    total_calls = 0
    while tree == None:
        epsilon = epsilon / 2.0
        tree, calls = Attempt(epsilon)
        total_calls += calls

    edges = tuple(sorted((b, a) for a, b in enumerate(tree) if b is not None))

    wilson_tree = nx.DiGraph()
    for u, v in edges:
        weight = (G[u][v]['weight'])
        wilson_tree.add_edge(u, v, weight=torch.log(weight))
    log_g = torch.tensor(0.0)
    # for u, v in edges:
    #     log_g += calculate_log_g(u, v, G_no_self_loops)
    return wilson_tree, edges, total_calls, log_g

def calculate_log_g(a, b, G):
    #summing log weights of all arcs in G
    weight_with_arc = torch.tensor([(G[u][v]['weight']) for (u, v) in G.edges()]).sum()
    # summing log weights of all arcs in G except for the given edge (a, b)
    weight_without_arc = torch.tensor([(G[u][v]['weight']) for (u, v) in G.edges() if (u, v) != (a, b)]).sum()

    theta = torch.exp(weight_with_arc - torch.log(torch.exp(weight_with_arc) + torch.exp(weight_without_arc)))
    log_g = torch.log(theta)
    return log_g