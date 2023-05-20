from sklearn.neighbors import kneighbors_graph
import numpy as np
# from mydataset import loadMvSlDataFromMat 
def generateMvG(X,k=5):
    print("data numbers:",len(X[0]))
    MvG = []
    for x in X:
        subG = kneighbors_graph(x,n_neighbors=k,n_jobs=-1,include_self=False,mode='distance')
        subG = subG.toarray()
        
        L = []
        for i,g in enumerate(subG):
            b=g
            sort = sorted(enumerate(b), key=lambda x:x[1])
            ind = [x[0] for x in sort]
            row = np.concatenate(([i],ind[-k:]))

            L.append(row)
        MvG.append(L)
    return np.array(MvG)

def getMvKNNGraph(X,k=5,mode='connectivity'):

    MvG = []
    for x in X:
        subG = kneighbors_graph(x,n_neighbors=k,n_jobs=-1,include_self=False,mode=mode)
        subG = subG.toarray()

        MvG.append(subG)
    return np.array(MvG)
# def getIncMvKNNGraph(X,inc_V_ind,k=5,mode='distance'):
#     print("data numbers:",len(X[0]))
#     MvG = []
#     for x in X:
#         subG = kneighbors_graph(x,n_neighbors=k,n_jobs=-1,include_self=False,mode=mode)
#         subG = subG.toarray()
        
#         MvG.append(subG)
#     return np.array(MvG)

if __name__ == '__main__':
    # mv_data,label = loadMvDataFromMat('/disk1/lcl/MATLAB-NOUPLOAD/cluster-data/data/bbcsport.mat',mode='nxd')
    # print(mv_data.shape,label.shape)
    
    a=[[[1],[3],[5],[2],[6]]]
    import torch
    a=torch.tensor(a)
    MvG = getMvKNNGraph(a,k=2)
    print(MvG)