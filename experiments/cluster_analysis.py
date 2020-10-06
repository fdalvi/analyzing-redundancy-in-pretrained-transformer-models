
import json
import numpy as np
from scipy.stats import mode

import matplotlib.pyplot as plt

tasks = ['SST-2', 'MRPC', 'MNLI', 'QNLI', 'QQP', 'RTE', 'STS-B']

def plot_cluster_windows(model):
    global_window_to_cluster = {}
    for task in tasks:
        res = json.load(open('<path-to-results-file-from-classification-experiments>/all_results.json'))
        clusters = res['selection']['clustering-0.30']['clusters']
        cluster_to_neuron = {cluster: [] for neuron, cluster in enumerate(clusters)}
        for neuron, cluster in enumerate(clusters):
            cluster_to_neuron[cluster].append(neuron)
        cluster_to_layers = {cluster: (np.array(cluster_to_neuron[cluster])/768).astype(np.int) for cluster in cluster_to_neuron}

        cluster_to_window = {cluster: np.max(layers) - np.min(layers) + 1 for cluster, layers in cluster_to_layers.items()}

        window_to_cluster = {}
        for cluster, window in cluster_to_window.items():
            if window not in window_to_cluster:
                window_to_cluster[window] = 0
            window_to_cluster[window] += 1
            
            if window not in global_window_to_cluster:
                global_window_to_cluster[window] = 0
            global_window_to_cluster[window] += 1
        print(task, window_to_cluster)
    
    print(global_window_to_cluster)
    x_axis = np.arange(1, max(global_window_to_cluster.keys())+1)
    print(x_axis)
    plt.figure()
    plt.bar(x_axis, [global_window_to_cluster.get(x,0) for x in x_axis])
plot_cluster_windows('bert')
plot_cluster_windows('xlnet')
