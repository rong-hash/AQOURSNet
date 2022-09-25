import numpy as np
import torch, os
from argparse import ArgumentParser
from warnings import filterwarnings
from utils import seed_torch, check_validity, get_eigenvalue, read_dataset, graph_dataloader
from construct_graph import extract_shapelets, embed_series, adjacency_matrix
from network import NeuralNetwork
from process import process, process_prf

filterwarnings('ignore')
parser = ArgumentParser(description='''AQOURSNet by Ziyuan Chen and Zhirong Chen.
See https://github.com/AllenHeartcore/AQOURSNet_rsch22su for the complete docs.''')

# PARAMS - DATA
parser.add_argument('dataset',     type=str,   help='Name of dataset')
parser.add_argument('--seed',      type=int,   default=42,       help='Random seed')
parser.add_argument('--device',    type=str,   default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

# PARAMS - SHAPELET & GRAPH
parser.add_argument('--nshapelet', type=int,   default=30,       help='Number of shapelets to extract')
parser.add_argument('--nsegment',  type=int,   default=20,       help='Number of segments for shapelet mapping')
parser.add_argument('--smpratio',  type=float, default=0,        help='Pos/Neg ratio for up/downsampling (0 = follow training set)')
parser.add_argument('--maxiter',   type=int,   default=300,      help='Max number of KMeans iterations')
parser.add_argument('--tol',       type=float, default=1e-4,     help='Tolerance of KMeans')
parser.add_argument('--percent',   type=int,   default=30,       help='Percentile for pruning weak edges')

# PARAMS - GAT
parser.add_argument('--dhidden',   type=int,   default=256,      help='Hidden dimension of GAT')
parser.add_argument('--dembed',    type=int,   default=64,       help='Embedding dimension of graph (output dim. of GAT)')
parser.add_argument('--nlayer',    type=int,   default=4,        help='Number of layers in GAT')
parser.add_argument('--nhead',     type=int,   default=8,        help='Number of attention heads in GAT')
parser.add_argument('--negslope',  type=float, default=.2,       help='Negative slope of leakyReLU')
parser.add_argument('--dropout',   type=float, default=.5,       help='Dropout rate')
parser.add_argument('--tail',      type=str,   default='linear', help='Type of prediction tail: [none, linear, mlp, resnet]')

# PARAMS - TRAINING
parser.add_argument('--nepoch',    type=int,   default=100,      help='Number of epochs')
parser.add_argument('--nbatch',    type=int,   default=16,       help='Number of mini-batches')
parser.add_argument('--optim',     type=str,   default='Adam',   help='Optimization algorithm for learning')
parser.add_argument('--lr',        type=float, default=.001,     help='Learning rate')
parser.add_argument('--wd',        type=float, default=.001,     help='Weight decay')
parser.add_argument('--amp',              action='store_true',   default=False,        help='Switch for using Automatic Mixed Precision')
parser.add_argument('--f1',               action='store_true',   default=False,        help='Switch for reporting F1 score in place of loss')

# PARAMS - ENHANCEMENTS
parser.add_argument('--ts2vec',           action='store_true',   default=False,        help='Switch for using TS2VEC')
parser.add_argument('--ts2vec-dhidden',   type=int,              default=64,           help='Hidden dimension of TS2Vec encoder')
parser.add_argument('--ts2vec-dembed',    type=int,              default=320,          help='Embedding dimension of TS2Vec encoder')
parser.add_argument('--ts2vec-nlayer',    type=int,              default=10,           help='Number of layers in TS2Vec encoder')
parser.add_argument('--dtw',              action='store_true',   default=False,        help='Switch for using Dynamic Time Warping')
parser.add_argument('--dtw-dist',         type=str,              default='euclidean',  help='Pointwise distance function of DTW')
parser.add_argument('--dtw-step',         type=str,              default='symmetric2', help='Local warping step pattern of DTW')
parser.add_argument('--dtw-window',       type=str,              default='none',       help='Windowing function of DTW')
parser.add_argument('--kmedians',         action='store_true',   default=False,        help='Switch for using KMedians in place of KMeans')

args = parser.parse_args()
check_validity(args, ['seed', 'smpratio', 'percent', 'negslope', 'dropout', 'wd'])
seed_torch(args.seed)

# DATA & PARAMS VERIFICATION
train_data, train_labels, test_data, test_labels = read_dataset(args.dataset)
(ntrain, lseries), (ntest, _) = train_data.shape, test_data.shape
setattr(args, 'nclass', train_labels.max() + 1)
setattr(args, 'nseries', ntrain)
setattr(args, 'lseries', lseries)
setattr(args, 'lshapelet', int(lseries / args.nsegment))
setattr(args, 'batchsize', len(train_data) // args.nbatch)
if args.device != 'cuda' and args.amp:
    print('WARNING: AMP is not supported for non-CUDA devices')
    args.amp = False
if args.nclass > 2:
    if args.smpratio != 0:
        print('WARNING: Up/downsampling is not supported for multi-class dataset')
        args.smpratio = 0
    if args.f1:
        print('WARNING: F1 score is not supported for multi-class dataset')
        args.f1 = False

# PARAMS PREPARATION
cwd = os.getcwd()
if not os.path.exists('%s/output' % cwd): os.mkdir('%s/output' % cwd)
setattr(args, 'nameset', args.dataset.split('/')[-1].split('.')[0])
for evtype, ext in zip(['shape', 'graph', 'model'], ['npy', 'npz', 'pt']):
    ev = get_eigenvalue(args, evtype)
    globals()['%s_eigenvalue' % evtype.lower()] = ev
    print('>>> Eigenvalue of %s = %s' % (evtype, ev))
    setattr(args, 'dir%s' % evtype, '%s/output/%s-%s-%s.%s' % (cwd, args.nameset, evtype, ev, ext))
setattr(args, 'dirlog', '%s/output/%s-log-%s.txt' % (cwd, args.nameset, model_eigenvalue))
if args.smpratio != 0:
    nshapelet_neg = int(args.nshapelet // (args.smpratio + 1))
    nshapelet_pos = args.nshapelet - nshapelet_neg
    print('\n>>> Statistics\n\tTrain set: %d * %d\n\tTest set:  %d * %d\n\tClasses:   %d\n\tShapes P:  %d * %d\n\tShapes N:  %d * %d\n'\
        % (ntrain, lseries, ntest, lseries, args.nclass, nshapelet_pos, args.lshapelet, nshapelet_neg, args.lshapelet))
else: print('\n>>> Statistics\n\tTrain set: %d * %d\n\tTest set:  %d * %d\n\tClasses:   %d\n\tShapes:    %d * %d\n'\
        % (ntrain, lseries, ntest, lseries, args.nclass, args.nshapelet, args.lshapelet))

# SHAPELET & GRAPH
if os.path.exists(args.dirgraph):
    print('[Loading Graph Cache]')
    graph = np.load(args.dirgraph)
    for field in graph.files:
        globals()[field] = graph[field]
else:
    if os.path.exists(args.dirshape):
        print('[Loading Shapelets Cache]')
        shapelets = np.load(args.dirshape)
    else:
        if args.smpratio == 0:
            shapelets = extract_shapelets(train_data, args.nshapelet, args, '')
        else:
            shapelets = np.r_[extract_shapelets(train_data[train_labels == 1], nshapelet_pos, args, 'Positive '),
                              extract_shapelets(train_data[train_labels == 0], nshapelet_neg, args, 'Negative ')]
        np.save(args.dirshape, shapelets)
    train_node_features = embed_series(train_data,  shapelets,  args, 'Train')
    test_node_features  = embed_series(test_data,   shapelets,  args, 'Test')
    train_edge_matrices = adjacency_matrix(train_node_features, args, 'Train')
    test_edge_matrices  = adjacency_matrix(test_node_features,  args, 'Test')
    np.savez(args.dirgraph,
            train_node_features=train_node_features, test_node_features=test_node_features,
            train_edge_matrices=train_edge_matrices, test_edge_matrices=test_edge_matrices)

# GAT & TRAINING
if os.path.exists(args.dirmodel):
    print('[Loading Pretrained Model]')
    model = torch.load(args.dirmodel).to(args.device)
else:
    model = NeuralNetwork(args).to(args.device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = eval('torch.optim.%s' % args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wd)
train_loader = graph_dataloader(train_node_features, train_edge_matrices, train_labels, args)
test_loader = graph_dataloader(test_node_features, test_edge_matrices, test_labels, args)
if args.f1: process_prf(model, train_loader, test_loader, loss_func, optimizer, args)
else: process(model, train_loader, test_loader, loss_func, optimizer, args)
