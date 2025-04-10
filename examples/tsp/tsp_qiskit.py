"""
This example explores the Traveling Salesman Problem (TSP) using QAOA with qiskit's interface.
"""

from itertools import permutations
import networkx as nx
import matplotlib.pyplot as plt
from qoptmodeler import QuantumTranslator, QAOASolver
import numpy as np
from pprint import pprint


def solve_tsp_qaoa(distance_matrix: np.ndarray) -> np.ndarray:
    print('Distance matrix:')
    pprint(distance_matrix)

    print('Starting mathematical formulation...')

    # Create matrices for the quadratic and linear terms -------------------------------------------------
    # For references see (docs/TSP problem explanation and mathematical formulation.pdf)
    # Cost function
    ones_with_zero_diag = np.ones((n, n))
    np.fill_diagonal(ones_with_zero_diag, 0)

    identity_with_one_row_permutated = np.eye(n)
    identity_with_one_row_permutated = np.roll(identity_with_one_row_permutated, 1, axis=0)

    cost_function = (
        np.kron(
            ones_with_zero_diag,
            (identity_with_one_row_permutated * distance_matrix)
        )
    )

    linear_cost_function = np.zeros(int(n ** 2))

    # Constraint matrices
    ones_n = np.ones(n)
    identity_n = np.eye(n)
    time_cons_matrix = np.kron(ones_n, identity_n)
    time_cons_vec = ones_n
    space_cons_matrix = np.kron(identity_n, ones_n)
    space_cons_vec = ones_n

    # Combine matrices in one
    constraints_lhs = np.vstack((time_cons_matrix, space_cons_matrix))
    constraints_rhs = np.hstack((time_cons_vec, space_cons_vec))

    # Get expected outputs ----------------------------------------------------------------------------
    all_perms = permutations(range(n))
    possible_ints = []
    for perm in all_perms:
        perm_mat = np.zeros((n, n), dtype=int)
        for i, j in enumerate(perm):
            perm_mat[i, j] = 1
        flattened_mat = perm_mat.flatten()
        int_repr = int("".join(map(str, flattened_mat)), 2)
        possible_ints.append(int_repr)

    # Map problem to a ISING problem ---------------------------------------------------------------------
    translator = QuantumTranslator(quad_cost_matrix=cost_function,
                                   lin_cost_matrix=linear_cost_function,
                                   lhs_eq_matrix=constraints_lhs,
                                   rhs_eq_vector=constraints_rhs)
    J, h = translator.to_ising()

    print('Mathematical formulation done.')
    # Solve problem with QAOA -----------------------------------------------------------------------------
    print('Solving problem with QAOA...')
    solver = QAOASolver(solver='qiskit', n_layers=2)
    solution = solver.solve(J, h, epochs=100, silent=False, possible_result_ints=possible_ints).reshape(n, n)
    print('Solving done. Solution:')
    pprint(solution)

    return solution


if __name__ == '__main__':
    # Define the TSP problem
    # Create a random symmetric distance matrix with "infinity" on the diagonal to avoid self loops
    n = 4
    INFINITE = 1e6
    distance_matrix = np.random.randint(low=1, high=100, size=(n, n))
    distance_matrix = distance_matrix + distance_matrix.T - np.diag(-INFINITE + np.diag(distance_matrix)).astype(int)

    # get solution
    solution = solve_tsp_qaoa(distance_matrix)


    # print solution
    def create_graph_from_distance_matrix(distance_matrix):
        n = distance_matrix.shape[0]
        G = nx.Graph()

        # Add nodes
        for i in range(n):
            G.add_node(i)

        # Add edges with weights
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] < INFINITE:
                    G.add_edge(i, j, weight=distance_matrix[i, j])

        return G


    # Draw the graph
    def draw_graph(G):
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, node_color='lightblue')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()


    # Create and draw the graph
    G = create_graph_from_distance_matrix(distance_matrix)
    draw_graph(G)
