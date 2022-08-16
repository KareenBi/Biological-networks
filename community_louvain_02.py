from __future__ import print_function

import array

import numbers
import warnings

import networkx as nx
import numpy as np
from regex import W

from community_status import Status

import networkx.algorithms.community as nx_comm
import sys                          #delete later
import matplotlib.cm as cm          #drawing graph
import matplotlib.pyplot as plt     #draw
from community import community_louvain
import leidenalg as la              #leiden algorithm
import igraph as ig                 #igraph for leiden algorithm

__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.


__PASS_MAX = -1
__MIN = 0.0000001

orig_stdout = sys.stdout
f =  open('out.txt', 'w') #!


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def partition_at_level(dendrogram, level):
    """Return the partition of the nodes at the given level
    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1.
    The higher the level is, the bigger are the communities
    Parameters
    ----------
    dendrogram : list of dict
       a list of partitions, ie dictionnaries where keys of the i+1 are the
       values of the i.
    level : int
       the level which belongs to [0..len(dendrogram)-1]
    Returns
    -------
    partition : dictionnary
       A dictionary where keys are the nodes and the values are the set it
       belongs to
    Raises
    ------
    KeyError
       If the dendrogram is not well formed or the level is too high
    See Also
    --------
    best_partition : which directly combines partition_at_level and
    generate_dendrogram : to obtain the partition of highest modularity
    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendrogram = generate_dendrogram(G)
    >>> for level in range(len(dendrogram) - 1) :
    >>>     print("partition at level", level, "is", partition_at_level(dendrogram, level))  # NOQA
    """
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph, weight='weight'):
    """Compute the modularity of a partition of a graph
    Parameters
    ----------
    partition : dict
       the partition of the nodes, i.e a dictionary where keys are their nodes
       and values the communities
    graph : networkx.Graph
       the networkx graph which is decomposed
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    Returns
    -------
    modularity : float
       The modularity
    Raises
    ------
    KeyError
       If the partition is not a partition of all graph nodes
    ValueError
        If the graph has no link
    TypeError
        If graph is not a networkx.Graph
    References
    ----------
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community
    structure in networks. Physical Review E 69, 26113(2004).
    Examples
    --------
    >>> import community as community_louvain
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(100, 0.01)
    >>> partition = community_louvain.best_partition(G)
    >>> modularity(partition, G)
    """
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def best_partition(graph,
                   partition=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_state=None):
    """Compute the partition of the graph nodes which maximises the modularity
    (or try..) using the Louvain heuristices
    This is the partition of highest modularity, i.e. the highest partition
    of the dendrogram generated by the Louvain algorithm.
    Parameters
    ----------
    graph : networkx.Graph
       the networkx graph which is decomposed
    partition : dict, optional
       the algorithm will start using this partition of the nodes.
       It's a dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    resolution :  double, optional
        Will change the size of the communities, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona
    randomize : boolean, optional
        Will randomize the node evaluation order and the community evaluation
        order to get different partitions at each call
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    partition : dictionnary
       The partition, with communities numbered from 0 to number of communities
    Raises
    ------
    NetworkXError
       If the graph is not undirected.
    See Also
    --------
    generate_dendrogram : to obtain all the decompositions levels
    Notes
    -----
    Uses Louvain algorithm
    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in
    large networks. J. Stat. Mech 10008, 1-12(2008).
    Examples
    --------
    >>> # basic usage
    >>> import community as community_louvain
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(100, 0.01)
    >>> partion = community_louvain.best_partition(G)
    >>> # display a graph with its communities:
    >>> # as Erdos-Renyi graphs don't have true community structure,
    >>> # instead load the karate club graph
    >>> import community as community_louvain
    >>> import matplotlib.cm as cm
    >>> import matplotlib.pyplot as plt
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> # compute the best partition
    >>> partition = community_louvain.best_partition(G)
    >>> # draw the graph
    >>> pos = nx.spring_layout(G)
    >>> # color the nodes according to their partition
    >>> cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    >>> nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
    >>>                        cmap=cmap, node_color=list(partition.values()))
    >>> nx.draw_networkx_edges(G, pos, alpha=0.5)
    >>> plt.show()
    """
    dendo = generate_dendrogram(graph,
                                partition,
                                weight,
                                resolution,
                                randomize,
                                random_state)
    return partition_at_level(dendo, len(dendo) - 1)


def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None):
    """Find communities in the graph and return the associated dendrogram
    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1. The higher the level is, the bigger
    are the communities
    Parameters
    ----------
    graph : networkx.Graph
        the networkx graph which will be decomposed
    part_init : dict, optional
        the algorithm will start using this partition of the nodes. It's a
        dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    resolution :  double, optional
        Will change the size of the communities, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona
    Returns
    -------
    dendrogram : list of dictionaries
        a list of partitions, ie dictionnaries where keys of the i+1 are the
        values of the i. and where keys of the first are the nodes of graph
    Raises
    ------
    TypeError
        If the graph is not a networkx.Graph
    See Also
    --------
    best_partition
    Notes
    -----
    Uses Louvain algorithm
    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in large
    networks. J. Stat. Mech 10008, 1-12(2008).
    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendo = generate_dendrogram(G)
    >>> for level in range(len(dendo) - 1) :
    >>>     print("partition at level", level,
    >>>           "is", partition_at_level(dendo, level))
    :param weight:
    :type weight:
    """
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    # Properly handle random state, eventually remove old `randomize` parameter
    # NOTE: when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_state = 0

    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = check_random_state(random_state)

    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = list()

    # started to change the code from here!
    __one_level(current_graph, status, weight, resolution, random_state)        # modularity optamisation
    
    # refinment step
    com2node = generate_com2node(status.node2com)
    for com in com2node:
        com_nodes = com2node[com]
        #f.write(str(len(com_nodes)))
        com_graph = current_graph.subgraph(com_nodes).copy()
        com_status = Status()
        com_status.init(com_graph, weight)
        one_level_refine(com_graph, com_status, weight ,resolution, random_state)
        com_partition = __renumber(com_status.node2com)
        update_status(current_graph, status, weight, com_partition)
    new_mod = __modularity(status, resolution)
    partition = __renumber(status.node2com)
    mod = new_mod 
    status_list.append(partition)
    current_graph = induced_graph(partition, current_graph, weight)             #community aggregation step
    status.init(current_graph, weight)
    
    while True:
        __one_level(current_graph, status, weight, resolution, random_state)    # modularity optamisation
        com2node = generate_com2node(status.node2com)                           # refinment step
        for com in com2node:
            com_nodes = com2node[com]
            com_graph = current_graph.subgraph(com_nodes).copy()
            com_status = Status()
            com_status.init(com_graph, weight)
            one_level_refine(com_graph, com_status, weight ,resolution, random_state)
            com_partition = __renumber(com_status.node2com)
            update_status(current_graph, status, weight, com_partition)
        new_mod = __modularity(status, resolution)
        if new_mod - mod < __MIN:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod 
        current_graph = induced_graph(partition, current_graph, weight)         #community aggregation step
        status.init(current_graph, weight) 
    
    return status_list[:]


def induced_graph(partition, graph, weight="weight"):
    """Produce the graph where nodes are the communities
    there is a link of weight w between communities if the sum of the weights
    of the links between their elements is w
    Parameters
    ----------
    partition : dict
       a dictionary where keys are graph nodes and  values the part the node
       belongs to
    graph : networkx.Graph
        the initial graph
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    Returns
    -------
    g : networkx.Graph
       a networkx graph where nodes are the parts
    Examples
    --------
    >>> n = 5
    >>> g = nx.complete_graph(2*n)
    >>> part = dict([])
    >>> for node in g.nodes() :
    >>>     part[node] = node % 2
    >>> ind = induced_graph(part, g)
    >>> goal = nx.Graph()
    >>> goal.add_weighted_edges_from([(0,1,n*n),(0,0,n*(n-1)/2), (1, 1, n*(n-1)/2)])  # NOQA
    >>> nx.is_isomorphic(ind, goal)
    True
    """
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})
    return ret


def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    """
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret


def load_binary(data):
    """Load binary graph as used by the cpp implementation of this algorithm
    """
    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph


def __one_level(graph, status, weight_key, resolution, random_state):
    """Compute one level of communities
    """
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution)
    new_mod = cur_mod

    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph.nodes(), random_state):
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
            neigh_communities = __neighcom(node, graph, status, weight_key)
            remove_cost = - neigh_communities.get(com_node,0) + \
                resolution * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status)
            best_com = com_node
            best_increase = 0
            for com, dnc in __randomize(neigh_communities.items(), random_state):
                incr = remove_cost + dnc - \
                       resolution * status.degrees.get(com, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_com = com
            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:
                modified = True
        new_mod = __modularity(status, resolution)
        if new_mod - cur_mod < __MIN:
            break


def __neighcom(node, graph, status, weight_key):
    """ Compute the communities in the neighborhood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def __remove(node, com, weight, status):
    """ Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def __insert(node, com, weight, status):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))


def __modularity(status, resolution):
    """ Fast compute the modularity of the partition of the graph using
    status precomputed
    """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree * resolution / links -  ((degree / (2. * links)) ** 2)
    return result


def __randomize(items, random_state):
    """Returns a List containing a random permutation of items"""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items

#---------------------------------------------------- ADDED FUNCTIONS ----------------------------------------------------
def generate_com2node(node2com):
    """Generating the dictionary com2node
    key: community the community's nodes
    """
    coms = set(node2com.values())
    com2node = {i:list() for i in coms}
    for node in node2com:
        com2node[node2com[node]].append(node)
    return com2node


def update_com2node(com2node, node, old_com, new_com):      #to delete! didn't use it eventually
    """Updating the com2node dictionaty after 
    moving node from old_com to new_com
    """
    com2node[old_com].remove(node)
    com2node[new_com].append(node)
    return com2node


def update_status(graph, status, weight_key, com_partition):
    """ updating the the current graph Status after the refinment step
    using __remove(), and __insert() functions
    """
    #n = len(set(status.node2com.values()))              # number of communities in the graph
    max_com = max(set(status.node2com.values())) + 1    # maximal community number
    com_n = len(set(com_partition.values()))            # number of communities in the subgraph
    if (com_n > 1): 
        #the community was splitted
        for node in com_partition:
            neigh_communities = __neighcom(node, graph, status, weight_key)
            old_com = status.node2com[node]
            new_com = max_com + com_partition[node]
            __remove(node, old_com, 
                    neigh_communities.get(old_com, 0.), status)
            __insert(node, new_com, 
                    neigh_communities.get(new_com, 0.), status)
    return 

        
def one_level_refine(graph, status, weight_key, resolution, random_state):
    """Refinement step
    each community is a connected graph
    graph: a community of the graph
    ...
    In iteration 1, len(community.nodes)==1 thus a connected graph
    in each iteration, we only move a node if the modularity improves 
    and all communities are still connected
    """
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution)
    new_mod = cur_mod
    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1
        #!!
        #com2node = generate_com2node(status.node2com)
        for node in __randomize(graph.nodes(), random_state):
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
            neigh_communities = __neighcom(node, graph, status, weight_key)
            remove_cost = - neigh_communities.get(com_node,0) + \
                resolution * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            com2node = generate_com2node(status.node2com)       #better to use update_com2node instead
                                                                #of generaring in every iteration
            com_nodes = set(com2node[com_node])-{node}
            subgraph = graph.subgraph(com_nodes).copy()
            if(len(subgraph.nodes())==0 or nx.is_connected(subgraph)):
                #removing node will keep it's old community connected
                __remove(node, com_node,
                        neigh_communities.get(com_node, 0.), status)
                best_com = com_node
                best_increase = 0
                for com, dnc in __randomize(neigh_communities.items(), random_state):
                    incr = remove_cost + dnc - \
                        resolution * status.degrees.get(com, 0.) * degc_totw
                    if incr > best_increase:
                        best_increase = incr
                        best_com = com
                __insert(node, best_com,
                        neigh_communities.get(best_com, 0.), status)
                if best_com != com_node:
                    #node changed it's community
                    modified = True
                    #!!
                    #update_com2node(com2node, node, com_node, best_com)
            new_mod = __modularity(status, resolution)
            if new_mod - cur_mod < __MIN:
                break


def generate_known_solution(path):
    """ Generating node2com dictionary based on the GeneOntology
    """
    node2com = {}
    with open(path) as f:
        for line in f:
            (key, val) = line.split("\t")
            node2com[key] = val.strip("\n")
    return node2com


def __jaccard(X, Y):
    """ Calculating the jaccard based on the lecture
    X: known solution, Y: our suggested solution
    """
    N_11 = 0    # i and j are assigned to the same cluster in X and Y
    N_00 = 0    # i and j are assigned to the different clusters in both X and Y
    N_10 = 0    # i and j are assigned to the same cluster in X but to different clusters in Y
    N_01 = 0    # i and j are assigned to the different clusters in X but the same in Y
    nodes = list(Y.keys())
    for i in range(len(nodes)):
        cluster_i_X = X[nodes[i]]
        cluster_i_Y = Y[nodes[i]]
        for j in range(i):
            cluster_j_X = X[nodes[j]]
            cluster_j_Y = Y[nodes[j]]
            if(cluster_i_X == cluster_j_X and cluster_i_Y == cluster_j_Y):
                N_11 += 1
            elif (cluster_i_X != cluster_j_X and cluster_i_Y != cluster_j_Y):
                N_00 += 1
            elif (cluster_i_X == cluster_j_X and cluster_i_Y != cluster_j_Y):
                N_10 += 1
            else:
                N_01 += 1
    jaccard = N_11/ (N_00 + N_10 + N_01)
    return jaccard


def run_code(cluster_path, edges_path, n):
    G = nx.read_edgelist(edges_path, delimiter = "\t")
    known_solution = generate_known_solution(cluster_path)
    jaccard_01, jaccard_02       = 0, 0
    modularity_01, modularity_02 = 0, 0
    for i in range(n):
        # calculations based on the original code
        partition_01 = community_louvain.best_partition(G)
        print(len(set(partition_01.values())))
        modularity_01 += modularity(partition_01, G)
        jaccard_01 += __jaccard(known_solution, partition_01)

        #calculations based on our extended code
        partition_02 = best_partition(G)
        print(len(set(partition_02.values())))
        modularity_02 += modularity(partition_02, G)
        jaccard_02 += __jaccard(known_solution, partition_02)
    
    modularity_01 = round(modularity_01/n, 5)
    jaccard_01 = round(jaccard_01/n, 5)
    modularity_02 = round(modularity_02/n, 5)
    jaccard_02 = round(jaccard_02/n, 5)
    return (modularity_01, jaccard_01, modularity_02, jaccard_02)


def run_leiden(edges_path, n=1):
    G = ig.Graph()
    G = ig.Graph.Read_Ncol(edges_path, directed = False)
    partition = la.find_partition(G, la.ModularityVertexPartition).modularity
    return partition


            

# ---------------------------------------------------------------------------------------------------------------------
yeast_edges = (r"/Users/kareen/Desktop/Semester_8/Biological_Networks/Benchmarks/Yeast/edges.txt")
yeast_cluster = (r"/Users/kareen/Desktop/Semester_8/Biological_Networks/Benchmarks/Yeast/clusters.txt")

arabidopsis_edges = (r"/Users/kareen/Desktop/Semester_8/Biological_Networks/Benchmarks/Arabidopsis/edges.txt")        
arabidopsis_cluster = (r"/Users/kareen/Desktop/Semester_8/Biological_Networks/Benchmarks/Arabidopsis/clusters.txt") 

n = 1
"""
result = run_leiden(yeast_edges)
print("Leiden result: ", result)
"""

modularity_01, jaccard_01, modularity_02, jaccard_02  = run_code(arabidopsis_cluster, arabidopsis_edges, n)

print("Jaccard_01: ", jaccard_01)
print("Modularity_01: ", modularity_01)
print("\n")

print("Jaccard_02: ", jaccard_02)
print("Modularity_02: ", modularity_02)
print("\n")

if(modularity_01>modularity_02):
    print("Original modularity is better :(")
elif(modularity_01<modularity_02):
    print("Our modularity is better ! :)")
else:
    print("same same..")

if(jaccard_01>jaccard_02):
    print("Original jaccard is better :(")
elif(jaccard_01<jaccard_02):
    print("Our jaccard is better ! :)")
else:
    print("same same..")


"""
# draw the graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
"""
