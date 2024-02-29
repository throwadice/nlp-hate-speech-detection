import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
#import pdb
for current_dataset in ['MD-Agreement']:  # loop on datasets
    for current_split in ['train']:  # loop on splits, here only train
        current_file = './' + current_dataset + '_dataset/' + current_dataset + '_' + current_split + '.json'  # current file
        train_data = json.load(open(current_file, 'r', encoding='UTF-8'))
        texts = []  # load data
        for item_id in train_data:  # loop across items for the loaded datasets
            text = train_data[item_id]['text']
            text = text.replace('\t', ' ').replace('\n', ' ').replace('\r',
                                                                      ' ')  # remove tabs and similar from text, so we can have everything on a line
            texts.append(text)
            # print('\t'.join([current_dataset, current_split, item_id, data[item_id]['lang'], str(data[item_id]['hard_label']), str(data[item_id]['soft_label']["0"]), str(data[item_id]['soft_label']["1"]), text]))
            # labeled_data.append((item_id, text))
for current_dataset in ['MD-Agreement']:  # loop on datasets
    for current_split in ['dev']:  # loop on splits, here only train
        current_file = './' + current_dataset + '_dataset/' + current_dataset + '_' + current_split + '.json'  # current file
        dev_data = json.load(open(current_file, 'r', encoding='UTF-8'))
        dev_texts = []  # load data
        for item_id in dev_data:  # loop across items for the loaded datasets
            text = dev_data[item_id]['text']
            text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            dev_texts.append(text)  # remove tabs and similar from text, so we can have everything on a line
#改进方案为先统计有多少个annatators？
#问题为：外侧第一个for 循环：我希望比较Ann418 和Ann266时，同时更新他们俩的值
# 初始化字典
result_dict = {}
#small_dic={}
# 遍历train_data中的所有样本
count=0
for n in train_data:
    if train_data[n]['hard_label']=="1" :
       count=count+1
a=0
for n in train_data:
    # 获取当前样本的annotators和annotations
    annotators = train_data[n]['annotators'].split(',')
    annotations = train_data[n]['annotations'].split(',')

    # 遍历每个annotator
    for i, annotator in enumerate(annotators):
        current_annotations = annotations[i]
        # 如果annotator不在result_dict中，则添加它
        if annotator not in result_dict:
            result_dict[annotator] = {}
            #j从1开始遍历？
            for j, other_annotators in enumerate(annotators):
                #other_annotators 不在小字典中，把它添加进去并且赋值为0
                other_annotations = annotations[j]
                if other_annotators not in result_dict[annotator]:
                 result_dict[annotator][other_annotators] = 0
                if current_annotations == other_annotations:
                       result_dict[annotator][annotators[j]] += 1
                    #不添加else得到dic：670，其中每个元素为dict：5
        else:
            for j,other_annotators in enumerate(annotators):
                other_annotations = annotations[j]
                if other_annotators not in result_dict[annotator]:
                    result_dict[annotator][other_annotators] = 0
                if current_annotations == other_annotations:
                    result_dict[annotator][annotators[j]] += 1


found_dicts = []  # 初始化一个列表，用于存储符合条件的字典

for i in result_dict:
    all_zero = True
    for j in result_dict[i]:
        if j != i and result_dict[i][j] != 0:
            all_zero = False
            break
    if all_zero:
        found_dicts.append(result_dict[i])  # 将符合条件的字典存储到列表中


import numpy as np

# 获取所有annatator的列表
all_annatators = list(result_dict.keys())

# 初始化一个670x670的array并将所有值设置为0
array = np.zeros((670, 670))

# 遍历result_dic中的每一个key和value
#annatators_list = [annatator for annatator in all_annatators ]
for i, key1 in enumerate(all_annatators):
    # 获取key1对应的value（即内部的dict）
    inner_dict = result_dict[key1]
    # 获取所有annatator的列表，以便按顺序填充array的行
    #annatators_list = [annatator for annatator in all_annatators ]
    # 遍历other_annatators，填充array的行
    for j, annatator in enumerate(all_annatators):
        # 如果annatator不在inner_dict中，将array的相应位置设置为-1e9
        if annatator not in inner_dict :
            array[i][j] = 0
        else:
            # 否则，将array的相应位置设置为inner_dict[annatator]的值
            array[i][j] = inner_dict[annatator]

# 将对角线上的元素设置为-1e9或0
np.fill_diagonal(array, 0)
# 数据预处理，移除全部为0的行和列
# 查找哪些行和列全为0
zero_rows = np.where(~array.any(axis=1))[0]
zero_cols = np.where(~array.any(axis=0))[0]

# 删除所有全为0的行和列
array = np.delete(array, zero_rows, axis=0)
array = np.delete(array, zero_cols, axis=1)

from sklearn.preprocessing import normalize

# 对相似度矩阵进行规范化，以确保每一行的值之和为1
array = normalize(array, axis=1, norm='l1')

# 将归一化后的矩阵与其转置矩阵取平均，得到对称矩阵
symmetric_array = (array + array.T) / 2
# 将小于某个阈值的相似度设置为0，以减少噪声和不必要的连接
#threshold = 0.1
#array[array < threshold] = 0
# 计算度矩阵
#degree_matrix = np.diag(np.sum(symmetric_array, axis=1))

# 构建标准拉普拉斯矩阵
#laplacian_matrix = degree_matrix - symmetric_array
# 进行特征分解，得到特征向量和特征值
eigenvalues, eigenvectors = np.linalg.eig(symmetric_array)

# 保留前k个特征向量，其中k是指定的降维后的维数
k = 2
selected_eigenvectors = eigenvectors[:, :k]
eigenvalues_sum = sum(eigenvalues)
eigenvalues_normalized = eigenvalues / eigenvalues_sum
selected_eigenvalues_normalized = eigenvalues_normalized[:k]
selected_eigenvalues_ratio = selected_eigenvalues_normalized / sum(selected_eigenvalues_normalized)
#Affinity_Matrix=array[np.diag_indices_from(array)] = -1e9
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering

#clustering = AffinityPropagation(damping=0.7).fit(array)
import matplotlib.pyplot as plt

# 调用 fit 方法训练聚类模型
#clustering = AffinityPropagation(damping=0.7).fit(array)
n_clusters = 6
#spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',n_init=100, assign_labels='kmeans')
spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',n_init=100, assign_labels='kmeans',random_state=1)
#spectral.fit(array)
# 可视化聚类结果
#labels = clustering.labels_
#n_clusters = len(set(labels))
#print(n_clusters)
# 选择一种颜色映射来绘制不同的聚类
#colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

#for i, color in zip(range(n_clusters), colors):
    # 选择属于第 i 个聚类的数据点
    #cluster = array[labels == i]
     #将该聚类的数据点绘制为散点图
    #plt.scatter(cluster[:, 0], cluster[:, 1], color=color)

#plt.title('Affinity Propagation Clustering')
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.show()
#a=train_data['1']['annotators']
labels=spectral.fit_predict(symmetric_array)
x=range(len(labels))
y=labels
scaler = MinMaxScaler()
selected_eigenvectors_normalized = scaler.fit_transform(selected_eigenvectors)
plt.scatter(x,y,c=labels)

plt.title('assign labels using k means')
plt.show()
cluster_annotators = {}
for i in range(n_clusters):
    cluster_indices = np.where(labels == i)[0]
    annotator_names = [all_annatators[idx] for idx in cluster_indices]
    cluster_annotators[f"Cluster {i}"] = annotator_names




def filter_train_data(train_data, cluster_annotators, cluster_key):
    # 获取指定的标注者集合
    a = cluster_annotators[cluster_key]

    # 存储符合条件的数据的列表
    selected_data = []

    # 遍历所有的训练数据
    for n, data in train_data.items():
        # 获取标注者集合
        b = data['annotators']
        b_set = set(b.split(','))

        # 判断b集合是否是a集合的子集
        if b_set.issubset(set(a)):
            selected_data.append(n)

    # 返回符合条件的数据列表
    return selected_data
Cluster0_data = filter_train_data(train_data, cluster_annotators, 'Cluster 0')
#['1', '3', '5', '8'...]
Cluster1_data = filter_train_data(train_data, cluster_annotators, 'Cluster 1')
Cluster2_data=filter_train_data(train_data, cluster_annotators, 'Cluster 2')
Cluster3_data=filter_train_data(train_data, cluster_annotators, 'Cluster 3')
Cluster4_data=filter_train_data(train_data, cluster_annotators, 'Cluster 4')
Cluster5_data=filter_train_data(train_data, cluster_annotators, 'Cluster 5')
