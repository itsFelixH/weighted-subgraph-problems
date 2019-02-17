import networkx as nx
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import graph_generator as gg
import graph_helper as gh
import weighted_subgraph_problem as wsp


# --------------------------
# PARAMETERS
# --------------------------

MODE = 'max'
DRAW = 1
ASK_USER = 1
CHOICE = 'a'

# --------------------------
# USER INPUT
# --------------------------


def main():

    if ASK_USER:
        print(text)
        choice = input('Your choice: ')
    else:
        choice = CHOICE
    
    if choice != 'z':
        
        if choice == 'a':
            G = gg.random_weighted_graph(10, 0.33, 10)
            start = timer()            
            (H, weight) = wsp.solve_full_ip(G, 'max')
            end = timer()
            
            if DRAW:
                ax = plt.subplot(1, 1, 1)
                dic = nx.spring_layout(G)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP', end-start)
                plt.show()
            
        elif choice == 'b':            
            G = gg.random_weighted_path(10, 10)
            dic = nx.circular_layout(G)
            
            start = timer()
            (H, weight) = wsp.solve_full_ip__rooted(G, MODE)
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 3, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (rooted)', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_full_ip(G, MODE)
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 3, 2)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_ip_on_path(G, MODE)
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 3, 3)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (path)', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_on_path__all_subpaths(G, MODE)
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 3, 4)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'subpath iteration', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_dynamic_prog_on_path(G, MODE)
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 3, 5)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'dynamic programm', end-start)

            # Results
            if DRAW:
                plt.show()

        elif choice == 'c':
            G = gg.random_weighted_path(500, 50)
            dic = nx.circular_layout(G)

            start = timer()
            (H, weight) = wsp.solve_on_path__all_subpaths(G, 'max')
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 1, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'iterating subpaths', end - start)

            start = timer()
            (H, weight) = wsp.solve_dynamic_prog_on_path(G, 'max')
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 1, 2)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'dynamic programm', end - start)

            #Results
            if DRAW:
                plt.show()
        
        elif choice == 'd':
            G = gg.random_weighted_binary_tree(10, 10)
            dic = nx.spring_layout(G)
            
            start = timer()
            (H, weight) = wsp.solve_full_ip__rooted(G, MODE)
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 2, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (rooted)', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_full_ip(G, MODE)
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 2, 2)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_on_tree__all_subtrees(G, MODE)
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 2, 3)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'subtree iteration', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_dynamic_prog_on_tree(G, MODE)
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 2, 4)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'dynamic programm', end-start)

            # Results
            if DRAW:
                plt.axis('off')
                plt.show()

        elif choice == 'e':
            G = gg.random_weighted_tree(100, 30)
            dic = nx.spring_layout(G)

            start = timer()
            (H, weight) = wsp.solve_separation(G, MODE)
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 1, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (separation)', end - start)

            start = timer()
            (H, weight) = wsp.solve_dynamic_prog_on_tree(G, MODE)
            end = timer()
            if DRAW:
                ax = plt.subplot(2, 1, 2)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP', end - start)

            # Results
            if DRAW:
                plt.axis('off')
                plt.show()


def draw_weighted_subgraph(plot, G, H, dic=None, weight=None, method='', time=None):
    if not dic:
        dic = nx.spring_layout()
    
    if not weight:
        weight = gh.weight(H)
    
    # Draw graph G
    nx.draw_networkx_nodes(G, pos=dic)
    nx.draw_networkx_edges(G, pos=dic, edge_color='r')
    
    # Draw subgraph H (blue)
    nx.draw_networkx_nodes(G, pos=dic, nodelist=H.nodes(), node_color='b')
    nx.draw_networkx_edges(G, pos=dic, edgelist=H.edges(), edge_color='b')
    
    # Label nodes and edges with weights
    edge_labels = dict(((u, v), str(d['weight'])) for u, v, d in G.edges(data=True))
    node_labels = dict((v, str(d['weight'])) for v, d in G.nodes(data=True))
    nx.draw_networkx_edge_labels(G, pos=dic, edge_labels=edge_labels)
    nx.draw_networkx_labels(G, pos=dic, labels=node_labels)
    
    # Label
    title = method + ': weight ' + str(weight)
    if time:
        title += ', time ' + str(round(time, 5)) + 's'
    plot.set_title(title)
    plot.axis('off')


text = '''
-------------------------------------------------------------------------------------
Algorithms for Weighted Subgraph Problems

  
Press a button.

a) Compute WSP on random weighted graph using an IP formulation
b) Compare WSP algorithms and their running times for paths
c) Compute WSP on big path
d) Compare WSP algorithms and their running times for trees
e) Compare IP (separation) and dynamic program for trees
      
z) End program...

'''

main()
