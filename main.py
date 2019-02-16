import gurobipy as gp
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
                dic = nx.spring_layout(G)
                draw_weighted_subgraph(G, H, dic, weight, 'an IP', end-start)
            
        elif choice == 'b':            
            G = gg.random_weighted_path(10, 10)
            dic = nx.circular_layout(G)
            
            start = timer()
            (H, weight) = wsp.solve_full_ip__rooted(G, MODE)
            end = timer()
            if DRAW:
                draw_weighted_subgraph(G, H, dic, weight, 'an IP (rooted)', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_full_ip(G, MODE)
            end = timer()
            if DRAW:
                draw_weighted_subgraph(G, H, dic, weight, 'an IP', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_ip_on_path(G, MODE)
            end = timer()
            if DRAW:
                draw_weighted_subgraph(G, H, dic, weight, 'an IP (path)', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_on_path__all_subpaths(G, MODE)
            end = timer()
            if DRAW:
                draw_weighted_subgraph(G, H, dic, weight, 'subpath iteration', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_dynamic_prog_on_path(G, MODE)
            end = timer()
            if DRAW:
                draw_weighted_subgraph(G, H, dic, weight, 'a dynamic programm', end-start)
        
        elif choice == 'c':            
            G = gg.random_weighted_binary_tree(10, 10)
            dic = nx.spring_layout(G)
            
            start = timer()
            (H, weight) = wsp.solve_full_ip__rooted(G, MODE)
            end = timer()
            if DRAW:
                draw_weighted_subgraph(G, H, dic, weight, 'an IP (rooted)', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_full_ip(G, MODE)
            end = timer()
            if DRAW:
                draw_weighted_subgraph(G, H, dic, weight, 'an IP', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_on_tree__all_subtrees(G, MODE)
            end = timer()
            if DRAW:
                draw_weighted_subgraph(G, H, dic, weight, 'subtree iteration', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_dynamic_prog_on_tree(G, MODE)
            end = timer()
            if DRAW:
                draw_weighted_subgraph(G, H, dic, weight, 'a dynamic programm', end-start)
        
        elif choice == 'd':
            G = gg.random_weighted_tree(6, 10)
            H = gh.direct_tree(G)
            
            nx.draw(G)
            plt.show()
            nx.draw(H)
            plt.show()

def draw_weighted_subgraph(G, H, dic=None, weight=None, method='', time=None):
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
    
    # Show plot
    plt.axis('off')
    title = 'Computed WSP (blue) with weight ' + str(weight) + ' using ' + method
    if time:
        title += ' in ' + str(round(time,3)) + 's'
    plt.title(title)
    plt.show()

    
text ='''
-------------------------------------------------------------------------------------
Algorithms for Weighted Subgraph Problems

  
Press a button.

a) Compute WSP on random weighted graph using an IP formulation
b) Compare WSP algorithms and their running times for paths
c) Compare WSP algorithms and their running times for trees
      
z) End program...

'''

main()