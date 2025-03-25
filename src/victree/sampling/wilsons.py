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
            Next[u] = RandomSuccessor(G, u)
            u = Next[u]
        u = i
        while not InTree[u]:
            InTree[u] = True
            u = Next[u]

    return Next

'''
INPUT: A graph G
OUTPUT: A random spanning tree, the edges, the total calls, and log_g
'''
def WilsonTree(G):
    n = len(G.nodes)

    def Attempt(epsilon):
        rand_predecessor_calls = 0
        Next = [None] * n
        InTree = [False] * n
        num_roots = 0
        log_g = torch.tensor(0.0)

        for i in range(n):
            u = i
            while not InTree[u]:
                if Chance(epsilon):
                    Next[u] = None
                    InTree[u] = True
                    num_roots += 1
                    if num_roots > 1:
                        return None, rand_predecessor_calls, log_g
                else:
                    rand_predecessor_calls += 1
                    Next[u], chosen_prob = RandomPredecessor(G, u)
                    log_g += torch.log(torch.tensor(chosen_prob))
                    u = Next[u]
            u = i
            while not InTree[u]:
                InTree[u] = True
                u = Next[u]

        return Next, rand_predecessor_calls, log_g

    G = AddSelfLoops(G)
    epsilon = 1.0
    tree = None

    total_calls = 0
    log_g = 0.0
    while tree == None:
        epsilon = epsilon / 2.0
        tree, calls, log_g_attempt = Attempt(epsilon)
        total_calls += calls
        log_g += log_g_attempt

    edges = tuple(sorted((b, a) for a, b in enumerate(tree) if b is not None))

    wilson_tree = nx.DiGraph()
    for u, v in edges:
        weight = G[u][v]['weight']
        wilson_tree.add_edge(u, v, weight=torch.tensor(torch.log(weight)))

    return wilson_tree, edges, total_calls, log_g




