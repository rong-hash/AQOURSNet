from ts2vec import TS2Vec
import datautils
from kmeans_pytorch import kmeans
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import torch, os, sys, random
from utils import init_dl_program

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *_):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def distance(series, shapelet):
    best_dist = np.inf
    for i in range(series.shape[0] - shapelet.shape[0] + 1):
        dist = np.linalg.norm(series[i:i+shapelet.shape[0]] - shapelet)
        if dist < best_dist:
            best_dist = dist
    return best_dist

def distance_matrix(series, shapelets):
    mat = np.zeros((series.shape[0], shapelets.shape[0]))
    for ns, s in tqdm(enumerate(series)):
        for nv, v in enumerate(shapelets):
            mat[ns, nv] = distance(s, v)
    return normalize(mat)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def extract_subsequence(series, sub_len):
    return np.concatenate([series[:, i:i+sub_len] for i in range(0, series.shape[1] - sub_len + 1, sub_len)])

def extract_shapelets_with_selfsupervised_learning(train_data, shapelets_num, shapelet_len):
    # Load the ECG200 dataset from UCR archive
    subsequence_data = extract_subsequence(train_data, shapelet_len)
    mean = np.nanmean(subsequence_data)
    std = np.nanstd(subsequence_data)
    subsequence_data = (subsequence_data - mean) / std
    subsequence_data_ = subsequence_data[..., np.newaxis]
    # (Both train_data and test_data have a shape of n_instances x n_timestamps x n_features)
    # Train a TS2Vec model
    device = init_dl_program(1, seed=None, max_threads=None)
    model = TS2Vec(
        input_dims=subsequence_data_.shape[2],
        device=device,
        output_dims=320
    )
    loss_log = model.fit(
        subsequence_data_,
        verbose=True,
        n_epochs = 20
    )


    # Compute instance-level representations for test set
    train_repr = model.encode(subsequence_data_, encoding_window='full_series')  # n_instances x output_dims
    with HiddenPrints():
        dis, classes = kmeans(torch.from_numpy(train_repr), num_clusters=shapelets_num, device=torch.device("cuda:1"))
    dis = dis.numpy().min(axis=1)
    shapelets = []
    for i in range(shapelets_num):
        shapelets.append(subsequence_data[classes == i][np.argmin(dis[classes == i])])
    shapelets = np.stack(shapelets)
    return shapelets





