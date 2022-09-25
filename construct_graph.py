import numpy as np
import torch
from tqdm import tqdm
from ts2vec import TS2Vec
from utils import pairwise_euc_dist, pairwise_dtw_dist, minmax_scale

def kmeans(points, ncluster, args, dtype):
    points = points.float().to(args.device)
    nsample = len(points)
    indices = np.random.choice(nsample, ncluster, replace=False)
    centers = points[indices]
    iteration = 0
    tqdm_meter = tqdm(desc='[Extracting %sShapelets]' % dtype)
    while iteration < args.maxiter:
        if args.dtw: dist = pairwise_dtw_dist(points, centers, args)
        else: dist = pairwise_euc_dist(points, centers)
        classes = torch.argmin(dist, dim=1)
        initial_state_pre = centers.clone()
        for index in range(ncluster):
            selected = torch.nonzero(classes == index).squeeze().to(args.device)
            selected = torch.index_select(points, 0, selected)
            if args.kmedians: centers[index] = selected.median(dim=0).values
            else: centers[index] = selected.mean(dim=0)
        center_shift = torch.sum(torch.sqrt(torch.sum( \
            (centers - initial_state_pre) ** 2, dim=1)))
        iteration += 1
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{args.tol:0.6f}')
        tqdm_meter.update()
        if center_shift ** 2 < args.tol: break
    return dist.cpu(), classes.cpu(), centers.cpu()

def extract_shapelets(series_set, nshapelet, args, dtype):
    cands = np.concatenate([series_set[:, i : i + args.lshapelet] \
                            for i in range(args.lseries - args.lshapelet + 1)])
    if args.ts2vec:
        if cands.ndim == 2: cands = np.expand_dims(cands, axis=2)
        model = TS2Vec(cands.shape[2], args.device,
                        hidden_dims=args.ts2vec_dhidden,
                        output_dims=args.ts2vec_dembed,
                        depth=args.ts2vec_nlayer)
        model.fit(cands)
        cands = model.encode(cands)
    dist, classes, centers = kmeans(torch.from_numpy(cands), nshapelet, args, dtype)
    if args.device == 'cuda': torch.cuda.empty_cache()
    if args.kmedians: return centers.cpu().numpy()
    dist = dist.cpu().numpy().min(axis=1)
    classes = classes.cpu().numpy()
    shapelets = []
    for i in range(nshapelet):
        shapelets.append(cands[classes == i][np.argmin(dist[classes == i])])
    return np.stack(shapelets)

def embed_series(series_set, shapelets, args, dtype):
    embedding = np.zeros((args.nseries, args.nshapelet, args.nsegment))
    residual = args.lseries % args.lshapelet
    for i, series in tqdm(enumerate(series_set), desc='[Embedding %s Series]' % dtype):
        if residual: series = series[:-residual]
        segments = series.reshape(-1, args.lshapelet)[:args.nsegment]
        if args.dtw: embedding[i] = pairwise_dtw_dist(shapelets, segments, args)
        else: embedding[i] = pairwise_euc_dist(shapelets, segments)
    return embedding

def adjacency_matrix(embedding, args, dtype):
    embedding = embedding.transpose(0, 2, 1)
    adj_mat = np.zeros((args.nseries, args.nshapelet, args.nshapelet))
    for ser_idx in tqdm(range(args.nseries), desc='[Constructing %s AdjMat]' % dtype):
        for seg_idx in range(embedding.shape[1] - 1):
            src_dist = 1. - minmax_scale(embedding[ser_idx, seg_idx])
            dst_dist = 1. - minmax_scale(embedding[ser_idx, seg_idx + 1])
            for src in range(args.nshapelet):
                adj_mat[ser_idx, src] += (src_dist[src] * dst_dist)
    threshold = np.percentile(adj_mat, args.percent)
    print('>>> %s AdjMat threshold = %f' % (dtype, threshold))
    return (adj_mat > threshold).astype(np.uint8)
