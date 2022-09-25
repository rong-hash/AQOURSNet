import numpy as np
from tqdm import tqdm
from kmeans_pytorch import kmeans
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import torch, os, random
from dtw import dtw

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
    dis, classes = kmeans(torch.from_numpy(candidates), num_clusters=shapelet_num, device=torch.device('cuda:2'))
    dis = dis.numpy().min(axis=1)
    shapelets = []
    for i in range(shapelet_num):
        shapelets.append(candidates[classes == i][np.argmin(dis[classes == i])])
    return np.stack(shapelets)

def distance(series, shapelet):
    subsequences = np.stack([series[i:i+len(shapelet)] for i in range(len(series) - len(shapelet) + 1)])
    distances = np.linalg.norm(subsequences - shapelet, axis=1)
    min = distances.min()
    new_distances = np.zeros(int(series.shape[0]/shapelet.shape[0]))
    for i, j in enumerate(range(0, distances.shape[0], shapelet.shape[0])):
        new_distances[i] = distances[j]
    return min, new_distances

def dtw_distance(series, shapelet):
    distance = np.zeros(len(series) - len(shapelet) + 1)
    for i in range(len(series) - len(shapelet) + 1):
        subsequence = series[i: i+len(shapelet)]
        match = dtw(shapelet, subsequence)
        distance[i] = match.distance
    new_distances = np.zeros(int(series.shape[0]/shapelet.shape[0]))
    for i, j in enumerate(range(0, distance.shape[0], shapelet.shape[0])):
        new_distances[i] = distance[j]
    return distance.min(), new_distances


def distance_matrix(series, shapelets):
    mat = np.zeros((series.shape[0], shapelets.shape[0]))
    distance_list = np.zeros((series.shape[0], shapelets.shape[0], int(series.shape[1]/shapelets.shape[1])))
    for ns, s in tqdm(enumerate(series)):
        for nv, v in enumerate(shapelets):
            mat[ns, nv], distance_list[ns, nv] = distance(s, v)
    series_shapelet_match = distance_list.argmin(1)        
    return normalize(mat), normalize(series_shapelet_match)

def dtw_distance_matrix(series, shapelets):
    mat = np.zeros((series.shape[0], shapelets.shape[0]))
    distance_list = np.zeros((series.shape[0], shapelets.shape[0], int(series.shape[1]/shapelets.shape[1])))
    for ns, s in tqdm(enumerate(series)):
        for nv, v in enumerate(shapelets):
            mat[ns, nv], distance_list[ns, nv] = dtw_distance(s, v)
    series_shapelet_match = distance_list.argmin(1)        
    return normalize(mat), normalize(series_shapelet_match)


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

# seed_torch()
# write_log('dataset,shapelet_num,shapelet_len,train_acc,test_acc\n')
# for dataset in datasets:
#     data = np.load('/data/chenzy/ucr/%s.npz' % dataset)
#     train_data, test_data = data['train_data'], data['test_data']
#     for shapelet_num in [20, 25, 30]:
#         for shapelet_len_prop in [.15, .2, .25, .3]:
            
#             shapelet_len = int(shapelet_len_prop * train_data.shape[1])
#             write_log('%s,%d,%d,' % (dataset, shapelet_num, shapelet_len))
#             shapelets = extract_shapelets(train_data, shapelet_num, shapelet_len)
            
#             _, train_mat = property_matrix(train_data, shapelets)
#             _, test_mat = property_matrix(test_data, shapelets)
#             classifier = SVC(random_state=0)
#             classifier.fit(train_mat, data['train_label'])
            
            # preds = classifier.predict(train_mat)
            # train_acc = (preds == data['train_label']).mean()
            # preds = classifier.predict(test_mat)
            # test_acc = (preds == data['test_label']).mean()
            
            # write_log('%f,' % train_acc)
            # write_log('%f\n' % test_acc)
            # print('%-24s\t%d\t%d\t%5.2f%%\t%5.2f%%' \
            #     % (dataset, shapelet_num, shapelet_len, train_acc * 100, test_acc * 100))
data = np.load('/data/chenzy/ucr/BeetleFly.npz')
seed_torch()
train_data, test_data = data['train_data'], data['test_data']
shapelet_num = 55
shapelet_len_prop = .2
shapelet_len = int(shapelet_len_prop * train_data.shape[1])
shapelets = extract_shapelets(train_data, shapelet_num, shapelet_len)
train_mat, train_mat_ = dtw_distance_matrix(train_data, shapelets)
test_mat, test_mat_ = dtw_distance_matrix(test_data, shapelets)
classifier = SVC(random_state=0)
classifier.fit(train_mat_, data['train_label'])
preds = classifier.predict(train_mat_)
train_acc = (preds == data['train_label']).mean()
preds = classifier.predict(test_mat_)
test_acc = (preds == data['test_label']).mean()
print("\n", train_acc, test_acc, "\n")



