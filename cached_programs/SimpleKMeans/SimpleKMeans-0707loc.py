import numpy as np
from tqdm import tqdm
from kmeans_pytorch import kmeans
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import torch, os, random

log_filename = 'SimpleKMeans-0707loc-exp.csv'
datasets = ['BeetleFly', 
            'Coffee', 
            'DistalPhalanxOutlineCorrect', 
            'Earthquakes', 
            'ECG200', 
            'ECGFiveDays', 
            'FordA', 
            'GunPoint', 
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

def extract_shapelets(series, shapelet_num, shapelet_len):
    candidates = np.concatenate([series[:, i:i+shapelet_len] \
                                for i in range(0, series.shape[1] - shapelet_len + 1)])
    dis, classes, _ = kmeans(torch.from_numpy(candidates), num_clusters=shapelet_num, device=torch.device('cuda'))
    dis = dis.numpy().min(axis=1)
    shapelets = []
    for i in range(shapelet_num):
        shapelets.append(candidates[classes == i][np.argmin(dis[classes == i])])
    return np.stack(shapelets)

def distance(series, shapelet):
    subsequences = np.stack([series[i:i+len(shapelet)] for i in range(len(series) - len(shapelet) + 1)])
    distances = np.linalg.norm(subsequences - shapelet, axis=1)
    return distances.min(), distances.argmin() / (len(series) - len(shapelet))

def property_matrix(series, shapelets):
    dist_mat = np.zeros((series.shape[0], shapelets.shape[0]))
    loc_mat = np.zeros((series.shape[0], shapelets.shape[0]))
    for ns, s in tqdm(enumerate(series)):
        for nv, v in enumerate(shapelets):
            dist_mat[ns, nv], loc_mat[ns, nv] = distance(s, v)
    return normalize(dist_mat), loc_mat

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
    with open(log_filename, 'a') as chart:
        chart.write(message)

seed_torch()
write_log('dataset,shapelet_num,shapelet_len,train_acc,test_acc\n')
for dataset in datasets:
    data = np.load('/data/chenzy/ucr/%s.npz' % dataset)
    train_data, test_data = data['train_data'], data['test_data']
    for shapelet_num in [20, 25, 30]:
        for shapelet_len_prop in [.15, .2, .25, .3]:
            
            shapelet_len = int(shapelet_len_prop * train_data.shape[1])
            write_log('%s,%d,%d,' % (dataset, shapelet_num, shapelet_len))
            shapelets = extract_shapelets(train_data, shapelet_num, shapelet_len)
            
            _, train_mat = property_matrix(train_data, shapelets)
            _, test_mat = property_matrix(test_data, shapelets)
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
