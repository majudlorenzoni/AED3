#importando bibliotecas
from time import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Funcões auxiliares
def read_matrix_file(file):
    with open(file) as matrix_file:
        matrix = [list(map(int, line.split())) for line in matrix_file]

    return matrix

def show_graph(matrix, draw_edges=False):
    G = nx.from_numpy_matrix(np.array(matrix))
    pos = nx.shell_layout(G)
    nx.draw(G, pos)

    if draw_edges:
        nx.draw_networkx_edge_labels(G, pos, label_pos=0.3)

    plt.show()

def path_to_matrix(path, matrix):
    # Creates an adjacency matrix representing the path
    nodes = range(len(matrix))
    path_matrix = np.zeros_like(matrix)

    for index in nodes:
        line = path[index]
        column = path[index + 1]

        edge_weight = matrix[line][column]
        path_matrix[line][column] = edge_weight
    
    return path_matrix

def calculate_path_cost(matrix, path):
    tsp_cost = 0
    nodes = range(len(matrix))

    for index in nodes:
        line = path[index]
        column = path[index + 1]

        edge_weight = matrix[line][column]

        tsp_cost += edge_weight

    return tsp_cost


#força bruta
def brute_force_tsp(matrix, path=[0], best_cost=float("inf"), best_path=None):
    # Recursion base
    if len(path) == len(matrix):
        # Path ends on the initial node
        path.append(0)
        final_cost = calculate_path_cost(matrix, path)

        if final_cost < best_cost:
            best_path = path.copy()
            best_cost = final_cost

        path.pop()

        return best_cost, best_path

    # Recursive step
    for node in range(len(matrix)):
        if node in path:
            continue

        path.append(node)

        best_cost, best_path = brute_force_tsp(matrix, path, best_cost, best_path)
        path.pop()

    return best_cost, best_path
# comparação

def show_results(matrix_file, run_brute_force):
    matrix = read_matrix_file(matrix_file)
    
    # Get cost from file name
    file_name = matrix_file.split('/').pop().upper()
    tsp, cost = file_name.split('_')
    cost = cost.split('.')[0]
    brute_force_time = '--'
    
    start_time = time()
    cost, path = brute_force_tsp(matrix)
    brute_force_time = time() - start_time
    
    return tsp, cost, brute_force_time

#arquivos
files = (("tsp1_253.txt", True),
         ("tsp2_1248.txt", False),
         ("tsp3_1194.txt", False),
         ("tsp4_7013.txt", False),
         ("tsp5_27603.txt", False))

print("TSP\t\tCusto FB\t\tTempo FB")

for tsp_file in files:
    tsp, brute_force = tsp_file
    
    # Alterar de acordo com a localização dos teus arquivos
    tsp, bf_cost, bf_time = show_results(f"C:/tsp_trabalho/tsp_exato/tsp_data/{tsp}", run_brute_force=brute_force)
    print(f'{tsp}\t\t{bf_cost}\t\t{bf_time:.5f}')
