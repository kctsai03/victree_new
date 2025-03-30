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


''' WIlSON'S ALGORITHM: HELPER FUNCTIONS '''

# Returns a copy of the graph with standardized indegree
# weights for each vertex.
def AddSelfLoops(G):
    G = G.copy()

    num_edges = G.size()
    max_weight = max([G[u][v]['weight'] for u, v in G.edges()])

    target_indegree = (G.number_of_nodes() - 1) * max_weight

    for node in G.nodes():
        self_loop_weight = target_indegree - G.in_degree(node, weight='weight')
        G.add_edge(node, node, weight=self_loop_weight)

    return G

# Returns a random successor vertex using the appropriate Markov chain
def RandomSuccessor(G, n):
    successors = list(G.successors(n))
    weights = [G[n][s]['weight'] for s in successors]

    return random.choices(successors, weights=weights, k=1)[0]

# Returns a random predecessor vertex using the appropriate Markov chain
def RandomPredecessor(G, n):
    predecessors = list(G.predecessors(n))
    weights = [G[s][n]['weight'] for s in predecessors]

    return random.choices(predecessors, weights=weights, k=1)[0]

# Chance(epsilon) returns true with probability epsilon.
def Chance(epsilon):
    return random.random() < epsilon