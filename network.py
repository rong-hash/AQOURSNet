import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GAT(nn.Module):
    def __init__(self, output_dim, args):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList(
            [GATv2Conv(args.nsegment,               args.dhidden, heads=args.nhead)] +
            [GATv2Conv(args.dhidden * args.nhead, args.dhidden, heads=args.nhead)] * (args.nlayer - 2) +
            [GATv2Conv(args.dhidden * args.nhead, output_dim, heads=1, concat=False)])
        self.neg_slope = args.negslope
        self.dropout = args.dropout
    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.leaky_relu(x, self.neg_slope)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return x

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=512, dropouts=[.1, .2, .2, .3], nlayer=4):
        super(MultilayerPerceptron, self).__init__()
        if isinstance(hidden_sizes, int): hidden_sizes = [hidden_sizes] * (nlayer - 1)
        if isinstance(dropouts, float): dropouts = [dropouts] * nlayer
        assert len(hidden_sizes) == nlayer - 1
        assert len(dropouts) == nlayer
        in_channels = [input_size] + hidden_sizes
        out_channels = hidden_sizes + [output_size]
        tails = [nn.ReLU()] * (nlayer - 1) + [nn.Softmax(dim=0)]
        layers = []
        for dropout, in_channel, out_channel, tail in zip(dropouts, in_channels, out_channels, tails):
            layers.extend([nn.Dropout(dropout), nn.Linear(in_channel, out_channel), tail])
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class FCResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_sizes=64, nlayer=3):
        super(FCResidualBlock, self).__init__()
        if isinstance(hidden_sizes, int): hidden_sizes = [hidden_sizes] * (nlayer - 1)
        assert len(hidden_sizes) == nlayer - 1
        in_channels = [input_size] + hidden_sizes
        out_channels = hidden_sizes + [input_size]
        layers = []
        for in_channel, out_channel in zip(in_channels, out_channels):
            layers.extend([nn.Linear(in_channel, out_channel), nn.ReLU()])
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x) + x

class FCResidualNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 128, 128], nblock=3, nlayer=3):
        super(FCResidualNetwork, self).__init__()
        if isinstance(hidden_sizes, int): hidden_sizes = [hidden_sizes] * nblock
        if isinstance(nlayer, int): nlayer = [nlayer] * nblock
        assert len(hidden_sizes) == nblock
        assert len(nlayer) == nblock
        blocks = []
        for (hidden_size, ilayer) in zip(hidden_sizes, nlayer):
            blocks.append(FCResidualBlock(input_size, hidden_size, ilayer))
        blocks.extend([nn.Linear(input_size, output_size), nn.Softmax(dim=0)])
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        return self.blocks(x)

class NeuralNetwork(nn.Module):
    def __init__(self, args):
        super(NeuralNetwork, self).__init__()
        if args.tail == 'none':
            self.gat = GAT(args.nclass, args)
            self.tail = None
        if args.tail == 'xgboost':
            raise NotImplementedError
        else:
            self.gat = GAT(args.dembed, args)
            if args.tail == 'linear':
                self.tail = nn.Linear(args.dembed, args.nclass)
            elif args.tail == 'mlp':
                self.tail = MultilayerPerceptron(args.dembed, args.nclass)
            elif args.tail == 'resnet':
                self.tail = FCResidualNetwork(args.dembed, args.nclass)
            else:
                raise NotImplementedError
    def forward(self, x, edge_index, batch):
        x = self.gat(x, edge_index, batch)
        if self.tail is not None: x = self.tail(x)
        return x
    def train(self):
        self.training = True
        self.gat.train()
        if self.tail is not None: self.tail.train()
    def eval(self):
        self.training = False
        self.gat.eval()
        if self.tail is not None: self.tail.eval()
