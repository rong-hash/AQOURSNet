import numpy as np
from tqdm import tqdm
from kmeans_pytorch import kmeans
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import torch, os, random

datasets = ['Computers', 
            'HandOutlines']

def extract_shapelets(series, shapelet_num, shapelet_len):
    candidates = np.concatenate([series[:, i:i+shapelet_len] \
                                for i in range(0, series.shape[1] - shapelet_len + 1)])
    dis, classes = kmeans(torch.from_numpy(candidates), num_clusters=shapelet_num, device=torch.device('cuda:2'))
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

def write_log(message):
    with open('SimpleKMeans_experiment.csv', 'a') as chart:
        chart.write(message)

seed_torch()
write_log('dataset,shapelet_num,shapelet_len,train_accuracy,test_accuracy\n')
for dataset in datasets:
    data = np.load('/data/chenzy/ucr/%s.npz' % dataset)
    train_data, test_data = data['train_data'], data['test_data']
    for shapelet_num in [10, 20, 30]:
        for shapelet_len_prop in [.05, .1, .2, .3]:
            
            shapelet_len = int(shapelet_len_prop * train_data.shape[1])
            write_log('%s,%d,%d,' % (dataset, shapelet_num, shapelet_len))
            shapelets = extract_shapelets(train_data, shapelet_num, shapelet_len)
            
            train_mat = distance_matrix(train_data, shapelets)
            test_mat = distance_matrix(test_data, shapelets)
            classifier = SVC(random_state=0)
            classifier.fit(train_mat, data['train_label'])
            
            preds = classifier.predict(train_mat)
            train_acc = (preds == data['train_label']).mean()
            preds = classifier.predict(test_mat)
            test_acc = (preds == data['test_label']).mean()
            
            write_log('%f,' % train_acc)
            write_log('%f\n' % test_acc)
            print('%-24s\t%d\t%d\t%5.2f%%\t%5.2f%%' \
                % (dataset, shapelet_num, shapelet_len, train_acc * 100, test_acc * 100))
