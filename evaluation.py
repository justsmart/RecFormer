import sklearn.metrics as metrics
import numpy as np
# from munkres import Munkres






def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment 
    ind = linear_assignment(w.max() - w)
    ind=np.asarray(ind)
    ind=np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def clustering_metric(y_true, y_pred, decimals=4):
    """Get clustering metric"""


    # ACC

    acc = cluster_acc(y_true, y_pred)
    acc = np.round(acc, decimals)
    # AMI
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    ami = np.round(ami, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)
    # PUR
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    pur = np.sum(np.amax(contingency_matrix,axis=0)) / np.sum(contingency_matrix)
    pur = np.round(pur, decimals)
    return dict({'ACC':acc,'AMI': ami, 'NMI': nmi, 'ARI': ari, 'PUR':pur})

## The following code is for function testing only and has nothing to do with the main code
if __name__ == '__main__':
    # a=np.array([1,1,0,1,2,1,2])
    # b=np.array([1,0,1,2,1,2,1])
    a = np.load('preds.npy').astype(int)
    b = np.load('label.npy').astype(int)
    print(clustering_metric(a,b)) 
    print(len(np.unique(a)))  
    # print(a,b)
