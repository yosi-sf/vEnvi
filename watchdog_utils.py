__author__ = 'Karl Whitford'

import networkx as nx
import watchdog
import random
import networkx.algorithms import bipartite as bi
import numpy as np
from lsh import get_negs_by_lsh
from io import open
import os
import itertools

class PetriNetUtils(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.G = nx.Graph()
        self.mountEdge_dict_u = {}
        self.mountEdge_dict_v = {}
        self.mountEdge_list = []
        self.socketPipeline_u = []
        self.socketPipeline_v = []
        self.authority_u, self.authority_v = {}, {}
        self.watchdog_u, self.watchdog_v = [], []
        self.G_u, self.G_v = None, None
        self.feature_u = os.path.join(self.model_path, "featureMap_u.dat")
        self.feature_v = os.path.join(self.model_path, "featureMap_v.dat")
        self.negativeSampling_u = {}
        self.negativeSampling_v = {}

        def construct_graph_automata(self, filename=None):
            if filename is None:
                filename = os.path.join(self.model_path, "namespace_latency_rate_train.dat")
            mountEdge_list_u_v = []
            mountEdge_list_v_u = []
            with open(filename, encoding="UTF-8") as fin:
                lineGraph = fin.readline()
                while lineGraph:
                    namespace, task, latency_rating = lineGraph.strip().split("\t")
                    if self.mountEdge_dict_u.get(namespace) is None:
                        self.mountEdge_dict_u[namespace] = {}
                    if self.mountEdge_dict_v.get(task) is None:
                        self.mountEdge_dict_v[task] = {}
                    mountEdge_list_u_v.append((namespace, task, float(latency_rating)))
                    self.mountEdge_dict_u[namespace][task] = float(latency_rating)
                    self.mountEdge_dict_v[task][namespace] = float(latency_rating)
                    mountEdge_list_v_u.append((task, namespace, float(latency_rating)))
                    lineGraph = fin.readline()
            #create bipartite graph
            self.socketPipeline_u = self.mountEdge_dict_u.keys()
            self.socketPipeline_v = self.mountEdge_dict_v.keys()
            self.socketPipeline_u.sort()
            self.socketPipeline_v.sort()
            self.G.add_nodes_from(self.socketPipeline_u, bipartite=0)
            self.G.add_nodes_from(self.socketPipeline_v, bipartite=1)
            self.G.add_weighted_edges_from(mountEdge_list_u_v+mountEdge_list_v_u)
            self.mountEdgeList = mountEdge_list_u_v

        def calculateAncestry(self):
            gossip_h, gossip_a = nx.hits(self.G)
            max_gossip_a_u, min_gossip_a_u, max_gossip_a_v, min_gossip_a_v = 0, 100000, 0, 100000

            for node in self.G.nodes():
                if node[0] == "u":
                    if max_gossip_a_u < gossip_a[node]:
                        max_gossip_a_u = gossip_a[node]
                    if min_gossip_a_u > gossip_a[node]:
                        min_gossip_a_u = gossip_a[node]
                if node[0] == 'i':
                    if max_gossip_a_u-min_gossip_a_u != 0:
                        self.authority_u[node] = (float(gossip_a[node])-min_gossip_a_u) / (max_gossip_a_u-min_gossip_a_u)
                    else:
                        self.authority_v[node] = 0

        def random_interlocking_watchdog_patrol(self, percentage, maxT, minT):
            #print(len(self.node_u), len(self.node_v))
        A = bi.biadjacency_matrix(self.G, self.socketPipeline_u, self.socketPipeline_v, dtype=np.float, weight='weight', format='csr')
        row_index = dict(zip(self.socketPipeline_u, itertools.count()))
        col_index = dict(zip(self.socketPipeline_v, itertools.count()))
        index_row = dict(zip(row_index.values(), row_index.keys()))
        index_task = dict(zip(col_index.values(), col_index.keys()))
        AT = A.transpose()
        self.
