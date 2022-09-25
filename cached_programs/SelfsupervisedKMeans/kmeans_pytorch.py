import numpy as np
import torch
from tqdm import tqdm

def initialize(X, num_clusters):
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        max_iter=300,
        tol=1e-4,
        device=torch.device('cpu')
):
    print(f'running k-means on {device}..')
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError
    X = X.float()
    X = X.to(device)
    initial_state = initialize(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    while iteration < max_iter:
        dis = pairwise_distance_function(X, initial_state)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)
        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        iteration = iteration + 1
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
    return dis.cpu(), choice_cluster.cpu()

def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    print(f'predicting on {device}..')
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError
    X = X.float()
    X = X.to(device)
    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)
    return choice_cluster.cpu()

def pairwise_distance(data1, data2, device=torch.device('cpu')):
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis

def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)
    cosine = A_normalized * B_normalized
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis
