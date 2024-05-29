import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATConv,  BatchNorm # noqa
import torch.nn as nn
from model.GNNs import *
import datetime
def main(S_name,max_epoch,isenhanced,pe,pf):
  
    import random
    if isenhanced=='True':
        graph = torch.load('../graphs/'+ S_name + '/enhanced_graph_'+str(pe)+'_'+str(pf)+ '.pt')
    else:
        graph = torch.load('../graphs/' + S_name + '/raw_graph.pt')
    print(graph)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = kanGCNNet(graph).to(device)
    data = graph.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch, eta_min=1e-4)

    model.train()
    best_train_acc=0
    best_vali_acc = 0
    best_test_acc = 0
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        out = model(graph)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
  
        model.eval()
        _, pred = model(graph).max(dim=1)
        correct_train = float(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
        acc_train = correct_train / data.train_mask.sum().item()
        if acc_train>best_train_acc:
            best_train_acc = acc_train

        _, pred = model(graph).max(dim=1)
        correct_val = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        acc_val = correct_val / data.val_mask.sum().item()
        if acc_val>best_vali_acc:
            best_vali_acc = acc_val

        _, pred = model(graph).max(dim=1)
        correct_test = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc_test = correct_test / data.test_mask.sum().item()
        if acc_test>best_test_acc:
            best_test_acc = acc_test
        print('Epoch: {:03d}, Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}, Test Acc: {:.4f}'.format(
            epoch+1, loss.item(), best_train_acc, best_vali_acc, best_test_acc))
        lr_schedule.step()
    return best_vali_acc,best_test_acc


import itertools
if __name__ == '__main__':
    for S_name in ['S1','S2','S3','S4']:
        for isenhanced in [False]:
            best_vali_acc,best_test_acc = main(S_name=S_name,max_epoch=200,isenhanced = False,pe=0.1,pf=0.1)
            np.save('../results/' + S_name + '/' + str(isenhanced) + '_best_vali_'+ '.npy', best_vali_acc)
            np.save('../results/' + S_name + '/' + str(isenhanced) + '_best_test_'+ '.npy', best_test_acc)


