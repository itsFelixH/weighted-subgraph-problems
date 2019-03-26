from timeit import default_timer as timer
import random
import os
import time
import networkx as nx
import matplotlib.pyplot as plt

import graph_generator as gg
import dynamic_program as dp
import integer_program as ip
import weighted_subgraph_problem as wsp
import heuristics as heu


def create_statistics_directory():
    directory = 'statistics'
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_statistics_file(name):
    create_statistics_directory()
    time_str = time.strftime("%Y%m%d-%H%M%S")
    file_name = 'times_' + name + '_' + time_str + '.csv'
    file_path = os.path.join('statistics', file_name)

    return file_path


def make_statistics(graph_class, iterations, sizes, stat_name=None, deliminator='&', mode='max', rooted=False,
                    full=False, flowrooted=False, flow=False, dyn=False, sep=False, sep_iter=False, span_heu=False,
                    set_heu=False):

    # weights
    weights = (-10, 10, -10, 10)

    if not stat_name:
        stat_name = graph_class
    file_path = create_statistics_file(stat_name)

    # Open file
    f = open(file_path, 'w')

    # Heading
    f.write(mode + '-WSP'+ deliminator + 'on' + deliminator + 'random' + deliminator + graph_class.capitalize()
            + 's' + '\n')
    f.write('Average' + deliminator + 'times' + deliminator + 'for' + deliminator + str(iterations) + deliminator
            + 'iterations:' + '\n')
    f.write('\n')

    table_columns = 'graph size'
    if dyn:
        table_columns += deliminator + 'dyn prog'
    if rooted:
        table_columns += deliminator + 'IP (rooted)'
    if full:
        table_columns += deliminator + 'IP'
    if flowrooted:
        table_columns += deliminator + 'IP (flow rooted)'
    if flow:
        table_columns += deliminator + 'IP (flow)'
    if set_heu:
        table_columns += deliminator + 'Node set'
    if span_heu:
        table_columns += deliminator + 'Spanning tree'
    if sep:
        table_columns += deliminator + 'IP(sep)'
    if sep_iter:
        table_columns += deliminator + 'iterations (sep)'
    table_columns += '\n'
    f.write(table_columns)

    # Fill table rows
    for n in sizes:
        print('Starting size ' + str(n) + ' at ' + time.strftime("%Y%m%d-%H%M%S"))

        times = dict()
        num_iter = []

        for k in range(iterations):
            if graph_class == 'path':
                G = gg.random_weighted_path(n, *weights)
            elif graph_class == 'tree':
                G = gg.random_weighted_tree(n, *weights)
            elif graph_class == 'SPG':
                size = 0
                while size != n:
                    G, D = gg.random_weighted_spg(2*n-3, *weights)
                    size = G.number_of_nodes()
                G = G.to_undirected()
            else:
                G = gg.random_connected_graph(n, int(1.1*n), *weights, multigraph=True)

            if (k + 1) % (iterations/10) == 0:
                print('Iteration ' + str(k + 1))

            if rooted:
                start = timer()
                (H, weight) = ip.solve_full_ip__rooted(G, mode)
                end = timer()
                alg = 'IP (rooted)'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if full:
                start = timer()
                (H, weight) = ip.solve_full_ip(G, mode)
                end = timer()
                alg = 'IP'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if flowrooted:
                start = timer()
                (H, weight) = ip.solve_flow_ip__rooted(G, mode)
                end = timer()
                alg = 'IP (flow rooted)'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if flow:
                start = timer()
                (H, weight) = ip.solve_flow_ip(G, mode)
                end = timer()
                alg = 'IP (flow)'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if set_heu:
                start = timer()
                H, weight_set = heu.node_set_heuristic(G, mode)
                end = timer()
                alg = 'Node set'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if span_heu:
                start = timer()
                H, weight_tree = heu.spanning_tree_heuristic(G, mode)
                end = timer()
                alg = 'Spanning tree'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if sep:
                start = timer()
                (H, weight, i) = ip.solve_separation_ip(G, mode)
                end = timer()
                alg = 'IP (separation)'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)
                num_iter.append(i)

            if dyn:
                if graph_class == 'path':
                    start = timer()
                    (H, weight) = dp.solve_dynamic_prog_on_path(G, mode)
                    end = timer()
                elif graph_class == 'tree':
                    start = timer()
                    (H, weight) = dp.solve_dynamic_prog_on_tree(G, mode)
                    end = timer()
                elif graph_class == 'SPG':
                    start = timer()
                    (H, weight) = dp.solve_dynamic_prog_on_spg(G, D, mode)
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
            table_row += deliminator + "{0:.6f}".format(av_time['Dynamic program'])
        if rooted:
            table_row += deliminator + "{0:.6f}".format(av_time['IP (rooted)'])
        if full:
            table_row += deliminator + "{0:.6f}".format(av_time['IP'])
        if flowrooted:
            table_row += deliminator + "{0:.6f}".format(av_time['IP (flow rooted)'])
        if flow:
            table_row += deliminator + "{0:.6f}".format(av_time['IP (flow)'])
        if span_heu:
            table_row += deliminator + "{0:.6f}".format(av_time['Spanning tree'])
        if set_heu:
            table_row += deliminator + "{0:.6f}".format(av_time['Node set'])
        if sep:
            table_row += deliminator + "{0:.6f}".format(av_time['IP (separation)'])
        if sep_iter:
            table_row += deliminator + "{0:.6f}".format(av_iter)

        table_row += '\\\\ \hline'
        table_row += '\n'
        f.write(table_row)

        print(graph_class + ' with size ' + str(n)
              + ' done' + ' at ' + time.strftime("%Y%m%d-%H%M%S"))
    f.close()

    print(graph_class + ' done' + ' at ' + time.strftime("%Y%m%d-%H%M%S"))


def make_weight_statistics(graph_class, iterations, weights, stat_name=None, deliminator='&', mode='max', rooted=False,
                           full=False, flowrooted=False, flow=False, dyn=False, sep=False):

    size = 100

    if not stat_name:
        stat_name = graph_class
    file_path = create_statistics_file(stat_name)

    # Open file
    f = open(file_path, 'w')

    # Heading
    f.write(mode + '-WSP'+ deliminator + 'on' + deliminator + 'random' + deliminator + graph_class.capitalize()
            + 's' + deliminator + 'of' + deliminator + 'size' + deliminator + str(size) + '\n')
    f.write('Average' + deliminator + 'times' + deliminator + 'for' + deliminator + str(iterations) + deliminator
            + 'iterations:' + '\n')
    f.write('\n')
    table_columns = 'Weights'
    if dyn:
        table_columns += deliminator + 'dyn prog'
    if rooted:
        table_columns += deliminator + 'IP (rooted)'
    if full:
        table_columns += deliminator + 'IP'
    if flowrooted:
        table_columns += deliminator + 'IP (flow rooted)'
    if flow:
        table_columns += deliminator + 'IP (flow)'
    if sep:
        table_columns += deliminator + 'IP(sep)'
    table_columns += '\n'
    f.write(table_columns)

    # Fill table rows
    for weight in weights:
        print('Starting weight ' + str(weight) + ' at ' + time.strftime("%Y%m%d-%H%M%S"))

        times = dict()

        for k in range(iterations):
            if graph_class == 'path':
                G = gg.random_weighted_path(size, *weight)
            elif graph_class == 'tree':
                G = gg.random_weighted_tree(size, *weight)
            elif graph_class == 'SPG':
                G, D = gg.random_weighted_spg(size, *weight)
            else:
                G = gg.random_connected_graph(size, 2*size, *weight)

            if (k + 1) % (iterations/10) == 0:
                print('Iteration ' + str(k + 1))

            if rooted:
                start = timer()
                (H, w) = ip.solve_full_ip__rooted(G, mode)
                end = timer()
                alg = 'IP (rooted)'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if full:
                start = timer()
                (H, w) = ip.solve_full_ip(G, mode)
                end = timer()
                alg = 'IP'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if flowrooted:
                start = timer()
                (H, w) = ip.solve_flow_ip__rooted(G, mode)
                end = timer()
                alg = 'IP (flow rooted)'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if flow:
                start = timer()
                (H, w) = ip.solve_flow_ip(G, mode)
                end = timer()
                alg = 'IP (flow)'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if sep:
                start = timer()
                (H, w, i) = ip.solve_separation_ip(G, mode)
                end = timer()
                alg = 'IP (separation)'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

            if dyn:
                if graph_class == 'path':
                    start = timer()
                    (H, weight) = dp.solve_dynamic_prog_on_path(G, mode)
                    end = timer()
                elif graph_class == 'tree':
                    start = timer()
                    (H, weight) = dp.solve_dynamic_prog_on_tree(G, mode)
                    end = timer()
                elif graph_class == 'SPG':
                    start = timer()
                    (H, weight) = dp.solve_dynamic_prog_on_spg(G, D, mode)
                    end = timer()

                alg = 'Dynamic program'
                if alg not in times:
                    times[alg] = []
                times[alg].append(end - start)

        av_time = dict()
        for alg in times:
            av_time[alg] = sum(times[alg]) / float(iterations)

        table_row = str(weight)

        if dyn:
            table_row += deliminator + "{0:.6f}".format(av_time['Dynamic program'])
        if rooted:
            table_row += deliminator + "{0:.6f}".format(av_time['IP (rooted)'])
        if full:
            table_row += deliminator + "{0:.6f}".format(av_time['IP'])
        if flowrooted:
            table_row += deliminator + "{0:.6f}".format(av_time['IP (flow rooted)'])
        if flow:
            table_row += deliminator + "{0:.6f}".format(av_time['IP (flow)'])
        if sep:
            table_row += deliminator + "{0:.6f}".format(av_time['IP (separation)'])

        table_row += '\\\\ \hline'
        table_row += '\n'
        f.write(table_row)

        print('weight ' + str(weight) + ' done' + ' at ' + time.strftime("%Y%m%d-%H%M%S"))
    f.close()

    print(graph_class + ' done' + ' at ' + time.strftime("%Y%m%d-%H%M%S"))


def make_relaxation_statistics(iterations, size, stat_name='relaxation', deliminator='&', mode='max', rooted=False,
                    full=False, flow=False):

    # weights
    weights = (-10, 10, -10, 10)
    file_path = create_statistics_file(stat_name)

    # Open file
    f = open(file_path, 'w')

    # Heading
    f.write(mode + '-WSP'+ deliminator + 'on' + deliminator + 'random' + deliminator + 'Graphs'
            + 's' + '\n')
    f.write('\n')
    table_columns = 'iteration' + deliminator + 'IP'
    if full:
        table_columns += deliminator + 'LP (full)' + deliminator + 'Gap (full)'
    if rooted:
        table_columns += deliminator + 'LP (rooted)' + deliminator + 'Gap (rooted)'
    if flow:
        table_columns += deliminator + 'LP (flow)' + deliminator + 'Gap (flow)'
    table_columns += '\n'
    f.write(table_columns)

    # Fill table rows

    for k in range(iterations):
        m = random.randint(size - 1, 30)
        G = gg.random_connected_graph(size, m, *weights)

        if (k + 1) % (iterations/10) == 0:
            print('Iteration ' + str(k + 1))

        (H, weight_ip) = ip.solve_flow_ip(G, mode)

        if rooted:
            start = timer()
            weight_rooted = ip.solve_full_ip__rooted(G, mode, relaxed=True)
            end = timer()
            alg = 'LP (rooted)'

        if full:
            start = timer()
            weight_full = ip.solve_full_ip(G, mode, relaxed=True)
            end = timer()
            alg = 'IP'

        if flow:
            start = timer()
            weight_flow = ip.solve_flow_ip(G, mode, relaxed=True)
            end = timer()
            alg = 'IP (flow)'

        table_row = str(int(weight_ip))
        if full:
            table_row += deliminator + "{0:.2f}".format(weight_full)
            table_row += deliminator + "{0:.2f}".format((1 - weight_ip/weight_full)*100) + '\%'
        if rooted:
            table_row += deliminator + "{0:.2f}".format(weight_rooted)
            table_row += deliminator + "{0:.2f}".format((1 - weight_ip/weight_rooted)*100) + '\%'
        if flow:
            table_row += deliminator + "{0:.2f}".format(weight_flow)
            table_row += deliminator + "{0:.2f}".format((1 - weight_ip/weight_flow)*100) + '\%'

        table_row += "\\\\ \hline \n"
        f.write(table_row)

    f.close()


def make_preprocessing_statistics(iterations, ns, stat_name='preprocess', deliminator='&', mode='max'):

    # weights
    weights = (-10, 10, -10, 10)

    file_path = create_statistics_file(stat_name)

    # Open file
    f = open(file_path, 'w')

    # Heading
    f.write(mode + '-WSP'+ deliminator + 'on' + deliminator + 'random' + deliminator + 'Graphs'
            + 's' + '\n')
    f.write('Average' + deliminator + 'times' + deliminator + 'for' + deliminator + str(iterations) + deliminator
            + 'iterations:' + '\n')
    f.write('\n')
    table_columns = 'n' + deliminator + 'm'
    table_columns += deliminator + 'with'
    table_columns += deliminator + 'n' + deliminator + 'm'
    table_columns += deliminator + 'without'
    table_columns += '\n'
    f.write(table_columns)

    # Fill table rows
    for n in ns:
        ms = [2*n, 3*n, 4*n]
        print('Starting size ' + str(n) + ' at ' + time.strftime("%Y%m%d-%H%M%S"))

        times = dict()

        #for k in range(iterations):
        for m in ms:
            G = gg.random_connected_graph(n, m, *weights, multigraph=True)
            R = wsp.preprocessing(G, mode)
            nr = R.number_of_nodes()
            mr = R.number_of_edges()

        # if (k + 1) % (iterations/10) == 0:
        #     print('Iteration ' + str(k + 1))
        #
        # start = timer()
        # (H, weight) = ip.solve_flow_ip(G, mode)
        # end = timer()
        # alg = 'No preprocessing'
        # if alg not in times:
        #     times[alg] = []
        # times[alg].append(end - start)
        #
        # start = timer()
        # (H, weight) = ip.solve_flow_ip(R, mode)
        # end = timer()
        # alg = 'Preprocessing'
        # if alg not in times:
        #     times[alg] = []
        # times[alg].append(end - start)
        #
        # av_time = dict()
        # for alg in times:
        #     av_time[alg] = sum(times[alg]) / float(iterations)

            table_row = str(n)
            table_row += deliminator + str(m)
            #table_row += deliminator + "{0:.6f}".format(av_time['Preprocessing'])
            table_row += deliminator + str(nr) + deliminator + str(mr)
            #table_row += deliminator + "{0:.6f}".format(av_time['No preprocessing'])

            table_row += '\\\\ \hline' + '\n'
            f.write(table_row)

        print('size ' + str(n) + ' done' + ' at ' + time.strftime("%Y%m%d-%H%M%S"))
    f.close()


def make_heuristic_statistics(iterations, sizes, stat_name='heuristic', deliminator='&', mode='max', set_heu=False,
                    span_heu=False):

    # weights
    weights = (-10, 10, -10, 10)
    file_path = create_statistics_file(stat_name)

    # Open file
    f = open(file_path, 'w')

    # Heading
    f.write(mode + '-WSP'+ deliminator + 'on' + deliminator + 'random' + deliminator + 'Graphs'
            + 's' + '\n')
    f.write('\n')
    table_columns = 'n'
    if set_heu:
        table_columns += deliminator + 'min gap' + deliminator + 'max gap' + deliminator + 'av gap'
    if span_heu:
        table_columns += deliminator + 'min gap' + deliminator + 'max gap' + deliminator + 'av gap'
    table_columns += '\\ \hline' + '\n'
    f.write(table_columns)

    # Fill table rows
    for n in sizes:
        min_gap_set = 100.0
        max_gap_set = 0.0
        min_gap_tree = 100.0
        max_gap_tree = 0.0

        gaps_set = []
        gaps_tree = []

        for k in range(iterations):
            size = 0
            while size != n:
                G, D = gg.random_weighted_spg(2 * n - 3, *weights)
                size = G.number_of_nodes()
            G = G.to_undirected()

            if (k + 1) % (iterations/10) == 0:
                print('Iteration ' + str(k + 1))

            G = G.to_undirected()
            (H, weight_ip) = dp.solve_dynamic_prog_on_spg(G, D, mode)

            if set_heu:
                H, weight_set = heu.node_set_heuristic(G, mode)
                gap_set = ((weight_ip - weight_set) / float(weight_ip))*100
                alg = 'Node set'
                gaps_set.append(gap_set)
                if gap_set > max_gap_set:
                    max_gap_set = gap_set
                if gap_set < min_gap_set:
                    min_gap_set = gap_set

            if span_heu:
                H, weight_tree = heu.spanning_tree_heuristic(G, mode)
                gap_tree = ((weight_ip - weight_tree) / float(weight_ip)) * 100
                alg = 'Spanning tree'
                gaps_tree.append(gap_tree)
                if gap_tree > max_gap_tree:
                    max_gap_tree = gap_tree
                if gap_tree < min_gap_tree:
                    min_gap_tree = gap_tree

        table_row = str(n)

        if set_heu:
            av_gap_set = sum(gaps_set) / float(iterations)

            table_row += deliminator + "{0:.2f}".format(max_gap_set)
            table_row += deliminator + "{0:.2f}".format(min_gap_set)
            table_row += deliminator + "{0:.2f}".format(av_gap_set)
        if span_heu:
            av_gap_tree = sum(gaps_tree) / float(iterations)

            table_row += deliminator + "{0:.2f}".format(max_gap_tree)
            table_row += deliminator + "{0:.2f}".format(min_gap_tree)
            table_row += deliminator + "{0:.2f}".format(av_gap_tree)
        table_row += "\\\\ \hline \n"
        f.write(table_row)

    f.close()
