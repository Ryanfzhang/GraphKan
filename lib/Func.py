import numpy as np
from torch_geometric.nn import GCNConv, ChebConv
import torch
import random
import torch_geometric
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from torch_geometric.data import DataLoader
from scipy.fftpack import fft, fftshift, ifft
import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy.fftpack import fftfreq
from dtaidistance import dtw
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
def calculate_dis(fea_matrix):
    dis_matrix =np.zeros([len(fea_matrix),len(fea_matrix)])
    for i in range(fea_matrix.shape[0]):
        for j in range(fea_matrix.shape[0]):
            if i == j :
                dis_matrix[i,j] = 0
            else:
                dis_matrix[i,j] = np.sqrt(np.sum(np.square(fea_matrix[i] - fea_matrix[j] )))
    return dis_matrix

def edge_construction_dtw(data_x, k_neighbor, isweight,isread):
    edge_index = [[], []]
    edge_weight = []
    if isread == False:
        dis = dtw.distance_matrix_fast(data_x)
        np.save('../features/edge/dis.npy', np.array(dis))
    else:
        dis=np.load('../features/edge/dis.npy')
    for j in range(0, len(data_x)):
        index = np.argsort(dis[j, :])
        a = np.where(index == j)
        index = np.delete(index, a, axis=0)
        for k in range(0, k_neighbor):
            edge_index[0].append(j)
            edge_index[1].append(index[k])
            if isweight == True and k_neighbor > 1:
                edge_weight.append(1.0 - k/k_neighbor)   #线性加权
                # edge_weight.append(1.0/(dis[j,index[k]])/(dis[j,index[0]]))
            else:
                edge_weight.append(1.0)  # 不加权
    return edge_index, edge_weight

def edge_construction_ed(data_x, k_neighbor, isweight,isread):
    edge_index = [[], []]
    edge_weight = []
    if isread == False:
        ed_dis = calculate_dis(data_x)
        np.save('../features/edge/ed_dis.npy', np.array(ed_dis))
    else:
        ed_dis=np.load('../features/edge/ed_dis.npy')
    start_num = 0
    for j in range(0, len(data_x)):
        index = np.argsort(ed_dis[j, :])
        a = np.where(index == j)
        index = np.delete(index, a, axis=0)
        for k in range(0, k_neighbor):
            edge_index[0].append(j + start_num)
            edge_index[1].append(index[k] + start_num)
            if isweight == True and k_neighbor > 1:
                edge_weight.append(1.0 - k/k_neighbor)   #线性加权
                # edge_weight.append(1.0/(dis[j,index[k]])/(dis[j,index[0]]))
            else:
                edge_weight.append(1.0)  # 不加权
    return edge_index, edge_weight


def normalize_samples(data):
    # 按列计算每个特征的均值和标准差
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # 对每个样本的每个特征进行归一化处理
    normalized_data = (data - mean) / std

    return normalized_data