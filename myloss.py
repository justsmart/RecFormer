import torch
import torch.nn as nn
import torch.nn.functional as F

def cosdis(x1,x2):
    return (1-torch.cosine_similarity(x1,x2,dim=-1))/2
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.t = 1
        self.tripletloss = nn.TripletMarginWithDistanceLoss(margin=1.0,distance_function=cosdis)

    def weighted_wmse_loss(self,input, target, weight, reduction='mean'):
        if isinstance(input,list):
            loss = [0]*len(input)
            for i in range(len(input)):
                loss[i] = torch.mean(weight[:,i:i+1].mul(target[i] - input[i]) ** 2)
            loss = torch.stack(loss,0)
        else:
            loss = (weight.unsqueeze(-1).mul(target - input)) ** 2

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        return loss

    def graph_loss(self,sub_graph,sub_x,all_x):
        if len(sub_graph.shape)==2:
     
            diag0_graph0 = torch.diag(sub_graph.sum(1))   # m*m for a m*n matrix 
            diag1_graph0 = torch.diag(sub_graph.sum(0))   # n*n for a m*n matrix              
            graph_loss = torch.trace(sub_x.t().mm(diag0_graph0).mm(sub_x))+torch.trace(all_x.t().mm(diag1_graph0).mm(all_x))-2*torch.trace(sub_x.t().mm(sub_graph).mm(all_x))
            return graph_loss/(sub_graph.shape[0]*sub_graph.shape[1])
        else:
            graphs_loss = 0
            for v,graph in enumerate(sub_graph):      
                diag0_graph0 = torch.diag(graph.sum(1))   # m*m for a m*n matrix 
                diag1_graph0 = torch.diag(graph.sum(0))   # n*n for a m*n matrix           
                graphs_loss += torch.trace(sub_x[v].t().mm(diag0_graph0).mm(sub_x[v]))+torch.trace(all_x[v].t().mm(diag1_graph0).mm(all_x[v]))-2*torch.trace(sub_x[v].t().mm(graph).mm(all_x[v]))
            return graphs_loss/(sub_graph.shape[0]*sub_graph.shape[1]*sub_graph.shape[2])
