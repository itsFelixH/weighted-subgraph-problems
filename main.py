import networkx as nx
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os
import time

import graph_generator as gg
import graph_helper as gh
import weighted_subgraph_problem as wsp
import dynamic_program as dp
import integer_program as ip
import statstics as stats
import heuristics as heu


# --------------------------
# PARAMETERS
# --------------------------

MODE = 'max'
ASK_USER = 1
CHOICE = 'a'
DRAW = 1
SAVE_PLOT = 0
PRINT_SOLUTION = 1

# For generating graphs
MIN_NODE_WEIGHT = -10
MAX_NODE_WEIGHT = 0
MIN_EDGE_WEIGHT = 0
MAX_EDGE_WEIGHT = 10
WEIGHTS = (-10, 10, -10, 10)

# For statistics
ITERATIONS = 100

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
            G = gg.random_connected_graph(200, 280, MIN_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

            start = timer()            
            (H, weight) = ip.solve_flow_ip(G, MODE)
            end = timer()

            if PRINT_SOLUTION:
                print('IP: weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(1, 1, 1)
                dic = nx.spring_layout(G)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (flow)', end - start)
            if SAVE_PLOT:
                dir = "SavedPlots"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                time_str = time.strftime("%Y%m%d-%H%M%S")
                name = 'Figure-' + time_str + '.png'
                file_path = os.path.join(dir, name)
                plt.savefig(file_path)
            if DRAW:
                plt.show()
            
        elif choice == 'b':            
            G = gg.random_weighted_path(12, MIN_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

            if DRAW or SAVE_PLOT:
                dic = nx.circular_layout(G)
                plt.rc('axes', titlesize=8)
                plt.rcParams['figure.figsize'] = 10, 5
                fig = plt.figure()
                fig.suptitle(MODE + '-WSP on a path')

            start = timer()
            (H, weight) = ip.solve_full_ip__rooted(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (rooted): weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 3, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (rooted)', end-start)
            
            start = timer()
            (H, weight) = ip.solve_full_ip(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP: weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 3, 2)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP', end-start)
            
            start = timer()
            (H, weight) = ip.solve_ip_on_path(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (path): weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 3, 3)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (path)', end-start)

            start = timer()
            (H, weight) = ip.solve_flow_ip(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (flow): weight ' + str(int(weight)) + ', time ' + str(round(end - start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 3, 4)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (flow)', end - start)
            
            start = timer()
            (H, weight) = wsp.solve_on_path__all_subpaths(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('subpath iteration: weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 3, 5)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'subpath iteration', end-start)
            
            start = timer()
            (H, weight) = dp.solve_dynamic_prog_on_path(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('dynamic program: weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 3, 6)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'dynamic program', end-start)

            # Results
            if SAVE_PLOT:
                dir = "SavedPathPlots"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                time_str = time.strftime("%Y%m%d-%H%M%S")
                name = 'Figure-' + time_str + '.png'
                file_path = os.path.join(dir, name)
                plt.savefig(file_path)
            if DRAW:
                plt.show()
        
        elif choice == 'c':
            G = gg.random_weighted_binary_tree(10, MIN_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

            if DRAW or SAVE_PLOT:
                dic = nx.spring_layout(G)
                plt.rc('axes', titlesize=8)
                plt.rcParams['figure.figsize'] = 10, 5
                fig = plt.figure()
                fig.suptitle(MODE + '-WSP on a tree')

            start = timer()
            (H, weight) = ip.solve_full_ip__rooted(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (rooted): weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 2, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (rooted)', end-start)
            
            start = timer()
            (H, weight) = ip.solve_full_ip(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP: weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 2, 2)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_on_tree__all_subtrees(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('subtree iteration: weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 2, 3)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'subtree iteration', end-start)
            
            start = timer()
            (H, weight) = dp.solve_dynamic_prog_on_tree(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('dynamic program: weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 2, 4)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'dynamic program', end-start)

            # Results
            if SAVE_PLOT:
                dir = "SavedTreePlots"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                time_str = time.strftime("%Y%m%d-%H%M%S")
                name = 'Figure-' + time_str + '.png'
                file_path = os.path.join(dir, name)
                plt.savefig(file_path)
            if DRAW:
                plt.axis('off')
                plt.show()

        elif choice == 'd':
            G = gg.random_weighted_path(400, MIN_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

            if DRAW:
                dic = nx.circular_layout(G)
                fig = plt.figure()
                fig.suptitle(MODE + '-WSP on a path')

            start = timer()
            (H, weight) = ip.solve_flow_ip(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (flow): weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 1, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (flow)', end - start)

            start = timer()
            (H, weight) = dp.solve_dynamic_prog_on_path(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('dynamic program: weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 1, 2)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'dynamic program', end - start)

            # Results
            if DRAW:
                plt.show()

        elif choice == 'e':
            G = gg.random_weighted_tree(60, MIN_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

            if DRAW:
                dic = nx.spring_layout(G)
                fig = plt.figure()
                fig.suptitle(MODE + '-WSP on a tree')

            start = timer()
            (H, weight, i) = ip.solve_separation_ip(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (sep): weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 1, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (' + str(i) + ' iterations)', end - start)

            start = timer()
            (H, weight) = dp.solve_dynamic_prog_on_tree(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('dynamic program: weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 1, 2)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'dynamic program', end - start)

            # Results
            if DRAW:
                plt.axis('off')
                plt.show()

        elif choice == 'f':
            G, D = gg.random_weighted_spg(10, MIN_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

            if DRAW or SAVE_PLOT:
                dic = nx.spring_layout(G)
                plt.rc('axes', titlesize=10)
                fig = plt.figure()
                fig.suptitle(MODE + '-WSP on a SPG')

            start = timer()
            (H, weight) = dp.solve_dynamic_prog_on_spg(G, D, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('dynamic program: weight ' + str(int(weight)) + ', time ' + str(round(end - start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 1, 1)
                draw_weighted_subgraph(ax, G.to_undirected(), H.to_undirected(), dic, weight, 'dynamic program', end - start)

            G = G.to_undirected()
            start = timer()
            (H, weight) = ip.solve_flow_ip(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (flow): weight ' + str(int(weight)) + ', time ' + str(round(end - start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 1, 2)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (flow)', end - start)

            # Results
            if SAVE_PLOT:
                dir = "SavedSPGPlots"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                time_str = time.strftime("%Y%m%d-%H%M%S")
                name = 'Figure-' + time_str + '.png'
                file_path = os.path.join(dir, name)
                plt.savefig(file_path)
            if DRAW:
                plt.axis('off')
                plt.show()

        elif choice == 'g':
            G, D = gg.random_weighted_spg(30, MIN_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

            G = G.to_undirected()
            dic_G = nx.spring_layout(G)

            R = wsp.preprocessing(G)

            # Draw graph G
            ax = plt.subplot(2, 2, 1)
            ax.axis('off')
            nx.draw_networkx_nodes(G, pos=dic_G, node_size=200)
            nx.draw_networkx_edges(G, pos=dic_G, edge_color='r', node_size=200)
            edge_labels = dict(((u, v), str(d['weight'])) for u, v, d in G.edges(data=True))
            node_labels = dict((v, str(d['weight'])) for v, d in G.nodes(data=True))
            nx.draw_networkx_edge_labels(G, pos=dic_G, edge_labels=edge_labels, font_size=8)
            nx.draw_networkx_labels(G, pos=dic_G, labels=node_labels, font_size=8)

            pos_higher = {}
            y_off = 0.2
            for k, v in dic_G.items():
                pos_higher[k] = (v[0], v[1] + y_off)
            labels = dict((v, v) for v in G.nodes)
            nx.draw_networkx_labels(G, pos_higher, labels)

            # Draw R
            ax = plt.subplot(2, 2, 2)
            ax.axis('off')
            dic_R = nx.spring_layout(R)
            nx.draw_networkx_nodes(R, pos=dic_R, node_size=200)
            nx.draw_networkx_edges(R, pos=dic_R, edge_color='r', node_size=200)
            edge_labels = dict(((u, v), str(d['weight'])) for u, v, d in R.edges(data=True))
            node_labels = dict((v, str(d['weight'])) for v, d in R.nodes(data=True))
            nx.draw_networkx_edge_labels(R, pos=dic_R, edge_labels=edge_labels, font_size=8)
            nx.draw_networkx_labels(R, pos=dic_R, labels=node_labels, font_size=8)

            pos_higher = {}
            y_off = 0.2
            for k, v in dic_R.items():
                pos_higher[k] = (v[0], v[1] + y_off)
            labels = dict((v, v) for v in R.nodes)
            nx.draw_networkx_labels(R, pos_higher, labels)

            start = timer()
            (H, weight) = ip.solve_flow_ip(G, MODE)
            print('weight: '+str(weight))
            print('weight: ' + str(gh.weight(H)))
            end = timer()
            if PRINT_SOLUTION:
                print('IP (flow): weight ' + str(int(weight)) + ', time ' + str(round(end - start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 2, 3)
                draw_weighted_subgraph(ax, G, H, dic_G, weight, 'IP (flow)', end - start)

            start = timer()
            (HR, weight) = ip.solve_flow_ip(R, MODE)
            print('weight: ' + str(weight))
            print('weight: ' + str(gh.weight(HR)))
            end = timer()
            if PRINT_SOLUTION:
                print('IP (flow): weight ' + str(int(weight)) + ', time ' + str(round(end - start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 2, 4)
                draw_weighted_subgraph(ax, R, HR, dic_R, weight, 'IP (flow)', end - start)

            plt.axis('off')
            plt.show()

        elif choice == 'h':
            verysmall = [5, 10, 11, 12, 13, 14, 15]
            small = [10, 20, 30, 40, 50, 60, 75, 100]

            # Statistics for comparing IPs
            stats.make_statistics('graph', ITERATIONS, verysmall, rooted=True, full=True, flowrooted=True, flow=True, sep=True)
            stats.make_statistics('graph', ITERATIONS, [20], full=True)
            stats.make_statistics('graph', ITERATIONS, [75, 100], flowrooted=True, flow=True, sep=True)

            # Statistics for dynamic
            stats.make_statistics('path', ITERATIONS [10, 20, 30, 40, 50, 60, 75, 100], stat_name='dyn_path', dyn=True,
                                 flow=True)
            stats.make_statistics('tree', ITERATIONS [10, 20, 30, 40, 50, 60, 75, 100], stat_name='dyn_path', dyn=True,
                                 flow=True)
            stats.make_statistics('SPG', ITERATIONS [50], stat_name='density', dyn=True, flow=True)

            # Statistics for comparing IPs
            stats.make_statistics('path', ITERATIONS, small, stat_name='dyn_path', dyn=True, flow=True)
            stats.make_statistics('path', ITERATIONS, small, mode='min', stat_name='dyn_path_min', dyn=True, flow=True)
            stats.make_statistics('path', ITERATIONS, small, stat_name='dyn_path', dyn=True, flow=True)
            stats.make_statistics('path', ITERATIONS, small, mode='min', stat_name='dyn_path_min', dyn=True, flow=True)
            stats.make_statistics('SPG', ITERATIONS, [1000], stat_name='density', flow=True, dyn=True)

            # Statistics for IP (sep)
            stats.make_statistics('graph', ITERATIONS, small, stat_name='sep', sep=True, sep_iter=True)

            # Statistics for preprocessing
            stats.make_preprocessing_statistics(ITERATIONS, [10, 20, 30, 40, 50, 60, 75, 100])
            stats.make_preprocessing_statistics(ITERATIONS, [200, 250, 500])

            # Statistics with GAP/Relaxing
            stats.make_relaxation_statistics(20, ITERATIONS, rooted=True, full=True, flow=True)


            # Statistic for weights
            weights = [(0, 0, -1, 1), (0, 0, -5, 5), (0, 0, -10, 10), (0, 0, -25, 25), (0, 0, -50, 50), (0, 0, -100, 100)]
            stats.make_weight_statistics('graph', ITERATIONS, weights, stat_name='weight', flow=True)
            weights = [(-1, 1, 0, 0), (-5, 5, 0, 0), (-10, 10, 0, 0), (-25, 25, 0, 0), (-50, 50, 0, 0), (-100, 100, 0, 0)]
            stats.make_weight_statistics('graph', ITERATIONS, weights, stat_name='weight', flow=True)
            weights = [(-10, 0, 0, 5), (-10, 0, 0, 10), (-10, 0, 0, 20), (-10, 0, 0, 25), (-10, 0, 0, 50), (-10, 0, 0, 100)]
            stats.make_weight_statistics('graph', ITERATIONS, weights, stat_name='weight', flow=True)

            # Heuristics
            stats.make_heuristic_statistics(ITERATIONS, 1000, span_heu=True)
            stats.make_heuristic_statistics(ITERATIONS, 100, set_heu=True)
            stats.make_statistics('graph', ITERATIONS, [10, 25, 50, 100, 250, 500, 1000], flow=True, span_heu=True, stat_name='heu_time')
            stats.make_statistics('graph', ITERATIONS, [10, 25, 50, 100, 250, 500, 1000], set_heu=True, stat_name='heu_time')

        elif choice == 'i':
            for k in range(10):
                print('Iteration ' + str(k+1))

                G = gg.random_connected_graph(10, 15, *WEIGHTS)

                start = timer()
                weight = ip.solve_flow_ip(G, MODE, relaxed=True)
                end = timer()
                if PRINT_SOLUTION:
                    print('LP (flow): weight ' + str(int(weight)) + ', time ' + str(round(end - start, 5)) + 's')

                start = timer()
                weight = ip.solve_full_ip__rooted(G, MODE, relaxed=True)
                end = timer()
                if PRINT_SOLUTION:
                    print('LP (rooted): weight ' + str(int(weight)) + ', time ' + str(round(end - start, 5)) + 's')

                start = timer()
                weight = ip.solve_full_ip(G, MODE, relaxed=True)
                end = timer()
                if PRINT_SOLUTION:
                    print('LP: weight ' + str(int(weight)) + ', time ' + str(round(end - start, 5)) + 's')

                start = timer()
                H, weight = ip.solve_flow_ip(G, MODE)
                end = timer()
                if PRINT_SOLUTION:
                    print('IP solution: weight ' + str(int(weight)) + ', time ' + str(round(end - start, 5)) + 's')

        elif choice == 'j':
            G, D = gg.random_weighted_spg(100, *WEIGHTS)
            G = G.to_undirected()

            H, weight = dp.solve_dynamic_prog_on_spg(G, D)
            print('weight OPT: '+str(int(weight)))
            HST, weight_st = heu.spanning_tree_heuristic(G)
            print('weight ST: '+str(int(weight_st)))
            HNS, weight_ns = heu.node_set_heuristic(G)
            print('weight NS: ' + str(int(weight_ns)))


def draw_weighted_subgraph(plot, G, H, dic=None, weight=None, method='', time=None):
    if not dic:
        dic = nx.spring_layout()
    
    if not weight:
        weight = gh.weight(H)
    
    # Draw graph G
    nx.draw_networkx_nodes(G, pos=dic, node_size=200)
    nx.draw_networkx_edges(G, pos=dic, edge_color='r', node_size=200)
    
    # Draw subgraph H (green)
    nx.draw_networkx_nodes(G, pos=dic, nodelist=H.nodes(), node_color='g', node_size=200)
    nx.draw_networkx_edges(G, pos=dic, edgelist=H.edges(), edge_color='g', node_size=200)
    
    # Label nodes and edges with weights
    edge_labels = dict(((u, v), str(d['weight'])) for u, v, d in G.edges(data=True))
    node_labels = dict((v, str(d['weight'])) for v, d in G.nodes(data=True))
    nx.draw_networkx_edge_labels(G, pos=dic, edge_labels=edge_labels, font_size=8)
    nx.draw_networkx_labels(G, pos=dic, labels=node_labels, font_size=8)
    
    # Label
    title = method + ': weight ' + str(int(weight))
    if time:
        title += ', time ' + str(round(time, 5)) + 's'
    plot.set_title(title)
    plot.axis('off')


text = '''
-------------------------------------------------------------------------------------
Algorithms for Weighted Subgraph Problems

  
Press a button.

a) Compute WSP on random weighted graph using IP (flow)
b) Compare WSP algorithms and their running times for paths
d) Compare WSP algorithms and their running times for trees
d) Compare IP (flow) and dynamic program for paths
e) Compare IP (separation) and dynamic program for trees
f) Compare IP (flow) and dynamic program for SPGs
g) Preprocessing for WSP on random graph
h) Save statistics to file
i) Compare relaxations
j) Compare heuristics
      
z) End program...

'''

main()
