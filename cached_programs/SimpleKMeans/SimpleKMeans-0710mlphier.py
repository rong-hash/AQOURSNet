import numpy as np
from tqdm import tqdm
from kmeans_pytorch import kmeans
from sklearn.preprocessing import normalize

import torch, os, random, warnings
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

######## TO BE MODIFIED ######## TO BE MODIFIED ######## TO BE MODIFIED ######## TO BE MODIFIED ########
########################################################################################################

# NETWORK STRUCTURE
class MLPBranch(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(MLPBranch, self).__init__()
        self.block = nn.Sequential(
                        nn.Dropout(.1), nn.Linear(input_size,  hidden_size), nn.ReLU(),
                        nn.Dropout(.2), nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                        nn.Dropout(.2), nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                        nn.Dropout(.3), nn.Linear(hidden_size, output_size), nn.Softmax(dim=1))
    def forward(self, x):
        return self.block(x)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, branch_num):
        super(NeuralNetwork, self).__init__()
        self.branches = []
        for _ in range(branch_num):
            self.branches.append(MLPBranch(input_size, input_size).to(device))
        self.tail = nn.Sequential(nn.Linear(input_size * branch_num, output_size), nn.Softmax(dim=1))
    def forward(self, x):
        inputs = torch.split(x, shapelet_num, dim=1)
        outputs = []
        for input, branch in zip(inputs, self.branches):
            output = branch(input)
            outputs.append(output)
        return self.tail(torch.cat(outputs, dim=1))

# HYPERPARAMETERS
shapelet_num = 50
shapelet_len_prop_range = [.1, .2, .3]
learning_rate = 1e-3
num_batch = 8
num_epoch = 100

# DATASET
log_filename = 'SimpleKMeans-0710mlphier-exp.csv'
datasets = ['BeetleFly', 
            'Coffee', 
            'DistalPhalanxOutlineCorrect', 
            'Earthquakes', 
            'ECG200', 
            'ECGFiveDays', 
            'FordA', 
            'Ham', 
            'ShapeletSim', 
            'SonyAIBORobotSurface1', 
            'SonyAIBORobotSurface2', 
            'Strawberry', 
            'ToeSegmentation1', 
            'TwoLeadECG', 
            'Wafer', 
            'WormsTwoClass', 
            'Yoga']

########### DANGER ZONE ########## DANGER ZONE ########## DANGER ZONE ########## DANGER ZONE ###########
########################################################################################################

class ShapeletEmbeddingDS(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_data(path):
    data = np.load(path)
    return (data[file] for file in data.files)

def get_dataloader(data, label, batch_num=num_batch):
    dataset = ShapeletEmbeddingDS(data, label)
    batch_size = len(dataset) // batch_num + 1
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def extract_shapelets(series, shapelet_num, shapelet_len):
    candidates = np.concatenate([series[:, i:i+shapelet_len] for i in range(0, series.shape[1] - shapelet_len + 1)])
    dis, classes = kmeans(torch.from_numpy(candidates), num_clusters=shapelet_num, device=torch.device(device))
    dis = dis.numpy().min(axis=1)
    shapelets = []
    for i in range(shapelet_num):
        shapelets.append(candidates[classes == i][np.argmin(dis[classes == i])])
    return np.stack(shapelets)

def distance(series, shapelet):
    subsequences = np.stack([series[i:i+len(shapelet)] for i in range(len(series) - len(shapelet) + 1)])
    distances = np.linalg.norm(subsequences - shapelet, axis=1)
    return distances.min()

def distance_matrix(series, shapelets):
    dist_mat = np.zeros((series.shape[0], shapelets.shape[0]))
    for ns, s in tqdm(enumerate(series)):
        for nv, v in enumerate(shapelets):
            dist_mat[ns, nv] = distance(s, v)
    return normalize(dist_mat)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def write_log(message):
    with open(log_filename, 'a') as log:
        log.write(message)

write_log('dataset,shapelet_num,shapelet_len,train_acc,test_acc\n')
warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_torch()

for dataset in datasets:
    shapelets_isvalid = True
    train_data, train_label, test_data, test_label = get_data('/data/chenzy/ucr/%s.npz' % dataset)
    if train_label.min() == 1: train_label -= 1
    if test_label.min() == 1: test_label -= 1

    shapelets = []
    for shapelet_len_prop in shapelet_len_prop_range:
        shapelet_len = int(shapelet_len_prop * train_data.shape[1])
        try:
            shapelets.append(extract_shapelets(train_data, shapelet_num, shapelet_len))
        except:
            shapelets_isvalid = False
            break
    if not shapelets_isvalid: continue

    write_log('%s,' % dataset)
    train_matrices, test_matrices = [], []
    for shapelet in shapelets:
        train_matrices.append(distance_matrix(train_data, shapelet))
        test_matrices.append(distance_matrix(test_data, shapelet))
    train_mat = np.concatenate(train_matrices, axis=1)
    test_mat = np.concatenate(test_matrices, axis=1)
    num_classes = train_label.max() + 1
    train_dataloader = get_dataloader(train_mat, train_label)

    model = NeuralNetwork(shapelet_num, num_classes, len(shapelets)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    best_train_acc, best_test_acc = 0, 0
    for epoch in range(num_epoch):
        train_acc = 0
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x.to(torch.float32))
            loss = loss_func(out, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc += (out.argmax(dim=1) == y).sum().item()
        train_acc /= len(train_label)
        with torch.no_grad():
            out = model(torch.Tensor(test_mat).to(device))
            test_acc = (out.argmax(dim=1).detach().cpu().numpy() == test_label).mean()
        print('epoch: %3d, loss: %8.6f, train_acc: %6.2f%%, test_acc: %6.2f%%' % (epoch + 1, loss, train_acc * 100, test_acc * 100))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if train_acc > best_train_acc: best_train_acc = train_acc

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

    write_log('%f,%f\n' % (best_train_acc, best_test_acc))
    print('%-24s\t%d\t%d\t%5.2f%%\t%5.2f%%' \
        % (dataset, shapelet_num, shapelet_len, best_train_acc * 100, best_test_acc * 100))