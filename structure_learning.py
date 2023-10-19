import pandas as pd
import networkx as nx
import sys
import time


def main():
    if len(sys.argv) != 2:
        raise Exception("usage: python project1.py <filename>")

    filename = sys.argv[1]

    # Load data
    data = pd.read_csv(f"./data/{filename}.csv")
    # Initialize graph
    G = nx.DiGraph()
    G.add_nodes_from(list(data.columns))

    start_time = time.time()

    nx.drawing.nx_pydot.write_dot(G, filename)

    end_time = time.time()

    print(f"start: {start_time} | end: {end_time} | time spent: {end_time - start_time}")


if __name__ == '__main__':
    main()





