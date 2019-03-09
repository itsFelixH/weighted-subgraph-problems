import networkx as nx
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os
import time

import graph_generator as gg
import graph_helper as gh
import weighted_subgraph_problem as wsp


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
MIN_NODE_WEIGHT = -200
MAX_NODE_WEIGHT = 0
MIN_EDGE_WEIGHT = 0
MAX_EDGE_WEIGHT = 100

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
            G = gg.random_weighted_graph(200, 0.01, MIN_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

            start = timer()            
            (H, weight) = wsp.solve_flow_ip(G, MODE)
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
            (H, weight) = wsp.solve_full_ip__rooted(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (rooted): weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 3, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (rooted)', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_full_ip(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP: weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 3, 2)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_ip_on_path(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (path): weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 3, 3)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (path)', end-start)

            start = timer()
            (H, weight) = wsp.solve_flow_ip(G, MODE)
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
            (H, weight) = wsp.solve_dynamic_prog_on_path(G, MODE)
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
            (H, weight) = wsp.solve_full_ip__rooted(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (rooted): weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW or SAVE_PLOT:
                ax = plt.subplot(2, 2, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (rooted)', end-start)
            
            start = timer()
            (H, weight) = wsp.solve_full_ip(G, MODE)
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
            (H, weight) = wsp.solve_dynamic_prog_on_tree(G, MODE)
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
            (H, weight) = wsp.solve_flow_ip(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (flow): weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 1, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (flow)', end - start)

            start = timer()
            (H, weight) = wsp.solve_dynamic_prog_on_path(G, MODE)
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
            (H, weight, i) = wsp.solve_separation_ip(G, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('IP (sep): weight ' + str(int(weight)) + ', time ' + str(round(end-start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 1, 1)
                draw_weighted_subgraph(ax, G, H, dic, weight, 'IP (' + str(i) + ' iterations)', end - start)

            start = timer()
            (H, weight) = wsp.solve_dynamic_prog_on_tree(G, MODE)
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
            G, D = gg.random_weighted_spg(60, MIN_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

            if DRAW or SAVE_PLOT:
                dic = nx.spring_layout(G)
                plt.rc('axes', titlesize=10)
                fig = plt.figure()
                fig.suptitle(MODE + '-WSP on a SPG')

            start = timer()
            (H, weight) = wsp.solve_dynamic_prog_on_spg(G, D, MODE)
            end = timer()
            if PRINT_SOLUTION:
                print('dynamic program: weight ' + str(int(weight)) + ', time ' + str(round(end - start, 5)) + 's')
            if DRAW:
                ax = plt.subplot(2, 1, 1)
                draw_weighted_subgraph(ax, G.to_undirected(), H.to_undirected(), dic, weight, 'dynamic program', end - start)

            G = G.to_undirected()
            start = timer()
            (H, weight) = wsp.solve_flow_ip(G, MODE)
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
            G, D = gg.random_weighted_spg(60, -20, 10, -10, 20)
            dic = nx.spring_layout(G)

            R, node_map, edge_map = wsp.preprocessing(G)

            # Draw graph G
            ax = plt.subplot(2, 1, 1)
            nx.draw(G)

            # Draw R
            ax = plt.subplot(2, 1, 2)
            nx.draw(R)
            plt.show()

        elif choice == 'h':
            small = [10, 20, 30, 40, 50, 60, 75, 100, 150, 200, 300, 400, 500]
            medium = [100, 250, 500, 1000, 2000]
            large = [500, 1000, 2500, 5000, 10000]

            #make_statistics('path', 10, large, sep=True, sep_iter=True)
            #make_statistics('tree', 10, large, sep=True, sep_iter=True)
            #make_statistics('spg', 10, small, sep=True, sep_iter=True)


def make_statistics(graph_class, iterations, sizes, mode='max', flow=False, dyn=False, sep=False, sep_iter=False):
    # Create directory
    dir = "statistics"
    if not os.path.exists(dir):
        os.makedirs(dir)
    time_str = time.strftime("%Y%m%d-%H%M%S")
    name = 'times_' + graph_class + '_' + time_str + '.csv'
    file_path = os.path.join(dir, name)

    # Open file
    f = open(file_path, 'w')

    # Heading
    f.write(mode + '-WSP,' + 'on' + ',' + graph_class.capitalize() + 's' + '\n')
    f.write('Average' + ',' + 'times' + ',' + 'for' + ',' + str(iterations) + ',' + 'iterations:' + '\n')
    f.write('\n')
    table_columns = 'graph size'
    if dyn:
        table_columns += '&' + 'time dynamic prog'
    if flow:
        table_columns += '&' + 'time IP (flow)'
    if sep:
        table_columns += '&' + 'time IP(sep)'
    if sep_iter:
        table_columns += '&' + 'IP(sep) iterations'
    table_columns += '\n'
    f.write(table_columns)

    # Fill table rows
    for n in sizes:
        print('Starting size ' + str(n) + ' at ' + time.strftime("%Y%m%d-%H%M%S"))

        times = dict()
        num_iter = []

        for k in range(iterations):
            if graph_class == 'path':
                G = gg.random_weighted_path(n, MAX_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)
            elif graph_class == 'tree':
                G = gg.random_weighted_tree(n, MAX_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)
            elif graph_class == 'spg':
                G, D = gg.random_weighted_spg(n, MAX_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)
            else:
                G = gg.random_weighted_graph(n, MAX_NODE_WEIGHT, MAX_NODE_WEIGHT, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

            if (k + 1) % (iterations/10) == 0:
                print('Iteration ' + str(k + 1))

            if flow:
                start = timer()
                (H, weight) = wsp.solve_flow_ip(G, mode)
                end = timer()
                alg = 'IP (flow)'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if sep:
                start = timer()
                (H, weight, i) = wsp.solve_separation_ip(G, mode)
                end = timer()
                alg = 'IP (separation)'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)
                num_iter.append(i)

            if dyn:
                start = timer()
                (H, weight) = wsp.solve_dynamic_prog_on_path(G, mode)
                end = timer()
                alg = 'Dynamic program'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

        av_time = dict()
        for alg in times:
            av_time[alg] = sum(times[alg]) / float(iterations)
        av_iter = sum(num_iter) / float(iterations)

        table_row = str(n)
        if dyn:
            table_row += "& {0:.6f}".format(av_time['Dynamic program'])
        if flow:
            table_row += "& {0:.6f}".format(av_time['IP (flow)'])
        if sep:
            table_row += "& {0:.6f}".format(av_time['IP (separation)'])
        if sep_iter:
            table_row += "& {0:.6f}".format(av_iter)
        table_row += '\n'
        f.write(table_row)

        print(graph_class + ' with size ' + str(n)
              + ' done' + ' at ' + time.strftime("%Y%m%d-%H%M%S"))
    f.close()

    print(graph_class + ' done' + ' at ' + time.strftime("%Y%m%d-%H%M%S"))


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
g) Preprocessing for GWSP on random graph
h) Save statistics to file
      
z) End program...

'''

main()
