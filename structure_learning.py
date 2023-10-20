import pandas as pd
import networkx as nx
import sys
import time
import numpy as np
from scipy.special import gammaln
import random
import matplotlib.pyplot as plt


def sub2ind(siz, x):
    k = np.concatenate(([1], np.cumprod(siz[:-1])))
    return int(np.dot(k, np.array(x) - 1)) + 1


def statistics(vars, G, D):
    n = len(vars)
    r = [var.r for var in vars]
    
    # Determine number of possible parent configurations for each variable
    q = [np.prod([r[j] for j in G.predecessors(i)]) for i in range(n)]
    
    # Create empty matrices
    M = [np.zeros((int(q[i]), int(r[i]))) for i in range(n)]
    
    for var_index in range(n):
        parents = list(G.predecessors(var_index))
        columns_of_interest = [vars[i].name for i in [var_index] + parents]
        
        grouped_data = D.groupby(columns_of_interest)["count"].sum().reset_index()
        
        for _, row in grouped_data.iterrows():
            k = int(row[vars[var_index].name]) - 1
            j = 0
            if parents:
                parent_values = [int(row[vars[p].name]) for p in parents]
                j = sub2ind([r[p] for p in parents], parent_values) - 1
            M[var_index][j, k] += row['count']

    return M


def prior(vars, G):
    n = len(vars)
    r = [vars[i].r for i in range(n)]
    q = [np.prod([r[j] for j in list(G.predecessors(i))]) for i in range(n)]
    return [np.ones((int(q[i]), int(r[i])), dtype=int) for i in range(n)]


def bayesian_score_component(M, alpha):
    p = np.sum(gammaln(alpha + M))
    p -= np.sum(gammaln(alpha))
    p += np.sum(gammaln(np.sum(alpha, axis=1)))
    p -= np.sum(gammaln(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p


def bayesian_score(vars, G, D):
    n = len(vars)
    M = statistics(vars, G, D)
    alpha = prior(vars, G)
    return sum(bayesian_score_component(M[i], alpha[i]) for i in range(n))


class Variable:
    def __init__(self, name, r):
        self.name = name
        self.r = r


def load_data(filename):
    df = pd.read_csv(f"./data/{filename}.csv", delimiter=',')
    df_max = df.max()
    var_names = list(df.columns)
    df = df.groupby(var_names).size().reset_index(name='count')
    vars = [Variable(var_names[i], df_max.iloc[i]) for i in range(len(var_names))]
    return df, vars


def load_graph(filename, vars):
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(vars))))
    names2idx = {vars[i].name: i for i in range(len(vars))}
    with open(f"./graphs/{filename}.gph", 'r') as f:
        for line in f:
            edge = line.replace('\n', '').replace(' ', '').split(',')
            G.add_edge(names2idx[edge[0]], names2idx[edge[1]])
    return G

def write_graph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write(f"{idx2names[edge[0]]}, {idx2names[edge[1]]}\n")

def plot_bayesian_network(bayesian_network, title, variable_names):
    pos = nx.spring_layout(bayesian_network)
    # Create a dictionary to map variable indices to their names
    variable_name_mapping = {i: variable_names[i] for i in range(len(variable_names))}
    # Replace node numbers with variable names in the plot
    labels = {node: variable_name_mapping[node] for node in bayesian_network.nodes()}
    nx.draw(bayesian_network, pos, labels=labels, with_labels=True, node_size=400, node_color="skyblue")
    plt.title(title)
    plt.show()

def test_score():
    filename = "example"
    # Load data
    df, vars = load_data(filename)
    G = load_graph(filename, vars)
    # Score graph
    expected_score = -132.57689402451837
    score = bayesian_score(vars, G, df)
    score_accuracy = 100 * (1 - (score - expected_score))
    print("Score accuracy: {:.0f}%".format(score_accuracy))


def random_network(vars, N_parents):
    G = nx.DiGraph()
    n = len(vars)
    
    for i in range(n):
        G.add_node(i)
    
    for i in range(n):
        nb_parents = random.randint(0, N_parents)
        if nb_parents > 0:
            parent_candidates = list(range(n))
            parent_candidates.remove(i)
            parents = random.sample(parent_candidates, nb_parents)
            for parent in parents:
                G.add_edge(parent, i)
    return G

# Create a global dictionary to stpre scores of Bayesian network structures
network_scores = {}

def select_best_BN(vars, D, n, N_parents):
    best_BN = None
    best_score = -float('inf')
    network_scores.clear()

    for i in range(n):
        initial_net = random_network(vars, N_parents)
        improved_net, score = k2(vars, D, N_parents)

        print(f"Iteration {i + 1}/{n} | Score: {score}")

        if score > best_score:
            best_BN = improved_net
            best_score = score

    return best_BN, best_score


def k2(vars, D, N_parents=None):
    n = len(vars)
    
    # Create an initial graph with no edges
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Variable ordering
    ordering = list(range(n))
    
    for i in ordering[1:]:
        y = bayesian_score(vars, G, D)
        
        parents_set = set(ordering[:i])
        current_parents = set(G.predecessors(i))
        
        if N_parents is not None and len(current_parents) >= N_parents:
            continue

        while True:
            y_best = -float("inf")
            j_best = None

            for j in parents_set:
                if not G.has_edge(j, i):
                    G.add_edge(j, i)
                    y_prime = bayesian_score(vars, G, D)
                    if y_prime > y_best:
                        y_best = y_prime
                        j_best = j
                    G.remove_edge(j, i)

            if j_best is not None and (N_parents is None or len(current_parents) < N_parents):
                G.add_edge(j_best, i)
                current_parents.add(j_best)
            else:
                break
    
    return G, y_best


def main():

    # Test Bayesian score
    test_score()

    small_df, small_vars = load_data("small")
    medium_df, medium_vars = load_data("medium")
    large_df, large_vars = load_data("large")

    small_vars_names = list(small_df.columns)
    medium_vars_names = list(medium_df.columns)
    large_vars_names = list(large_df.columns)

    # Small dataset
    small_idx2names = {i: small_vars_names[i] for i in range(len(small_vars_names))}
    start_time = time.time()
    small_best_BN, small_best_score = select_best_BN(small_vars, small_df, n=5, N_parents=2)
    end_time = time.time()
    print(f"Time spent: {end_time - start_time} seconds")
    print("Best Bayesian Score:", small_best_score)
    plot_bayesian_network(small_best_BN, "Bayesian Network - Small Dataset", small_vars_names)
    write_graph(small_best_BN, small_idx2names, "small")







if __name__ == '__main__':
    main()





