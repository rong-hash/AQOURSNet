import numpy as np
import torch, os, random
from crypt import crypt
from dtw import dtw
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# main.py

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_validity(args, allow_zero):
    for field, value in args.__dict__.items():
        if not (isinstance(value, str) or isinstance(value, bool)):
            if field in allow_zero:
                assert value >= 0, 'args.%s must be non-negative' % field
            else:
                assert value > 0, 'args.%s must be positive' % field

def get_eigenvalue(args, evtype):
    argsval = tuple(args.__dict__.values())
    argsval = [str(i) for i in argsval]
    if evtype == 'shape': argsval = '-'.join(argsval[3:8] + argsval[23:32])
    if evtype == 'graph': argsval = '-'.join(argsval[3:9] + argsval[23:32])
    if evtype == 'model': argsval = '-'.join(argsval[3:16] + argsval[23:32])
    salt = [str(args.seed)] * 8
    salt = '$1$' + ''.join(salt)
    eigenvalue = crypt(argsval, salt)[12:]
    return eigenvalue.replace('.', '_').replace('/', '+')

def read_dataset(dataset_name):
    try:
        data = np.load('%s.npz' % dataset_name)
    except FileNotFoundError:
        raise FileNotFoundError('Dataset %s not found.' % dataset_name)
    return (data[file] for file in data.files)

class GraphDataset(Dataset):
    def __init__(self, node_features, edge_matrices, labels):
        super(GraphDataset, self).__init__()
        node_features, edge_matrices, labels = map(lambda x: torch.tensor(x),
                                                   (node_features, edge_matrices, labels))
        self.data = []
        for node_feature, edge_matrix, label in zip(node_features, edge_matrices, labels):
            self.data.append(Data(x=node_feature.float(),
                                  edge_index=edge_matrix.nonzero().T,
                                  y=label, num_nodes=node_feature.shape[0]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def graph_dataloader(node_features, edge_matrices, labels, args):
    dataset = GraphDataset(node_features, edge_matrices, labels)
    return DataLoader(dataset, batch_size=args.batchsize,
                      shuffle=True, num_workers=torch.cuda.device_count())

# construct_graph.py

def pairwise_euc_dist(A, B):
    if isinstance(A, torch.Tensor):
        A = A.unsqueeze(dim=1)
        B = B.unsqueeze(dim=0)
        return ((A - B) ** 2).sum(dim=-1)
    else:
        A = np.expand_dims(A, axis=1)
        B = np.expand_dims(B, axis=0)
        return ((A - B) ** 2).sum(axis=-1)

def pairwise_dtw_dist(A, B, args):
    convert = False
    if isinstance(A, torch.Tensor):
        A, B = map(lambda x: x.cpu().numpy(), (A, B))
        convert = True
    dist = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            dist[i, j] = dtw(A[i], B[j], distance_only=True, dist_method=args.dtw_dist,
                             step_pattern=args.dtw_step, window_type=args.dtw_window).distance
    if convert: dist = torch.from_numpy(dist).to(args.device)
    return dist

def minmax_scale(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

# process.py

def write_log(log_filename, message):
    with open(log_filename, 'a') as log:
        log.write(message)

def prf_score(y_pred, y_true, epsilon=1e-7):
    y_pred = y_pred.argmax(dim=1)
    tp = (y_true * y_pred).sum().float()
    fp = ((1 - y_true) * y_pred).sum().float()
    fn = (y_true * (1 - y_pred)).sum().float()
    prec = tp / (tp + fp + epsilon)
    recl = tp / (tp + fn + epsilon)
    return prec * 100, recl * 100, (prec * recl) / (prec + recl + epsilon) * 200
