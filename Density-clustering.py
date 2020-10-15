import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix
import os
from pandas.api.types import is_string_dtype

def read_data(filenames):
    '''
    Input: filenames -- a dictionary contain data and names file names
    Output: data -- *.data file as a data frame with column names as that in *.names file
            names_dict -- *.names file as a dictionary of list containing the content that
                            can be present in the respective column in data file
    '''
    
    with open(filenames['names_file'], 'r') as temp_f:
        col_count = [ len(l.split(',')) for l in temp_f.readlines() ]
    column_names = [i for i in range(0, max(col_count)+1)]
    
    names = pd.read_csv(filenames['names_file'],
                        names = column_names,
                        skiprows = 1,
                        index_col = False,
                        skipinitialspace = True,
                        header = None,
                        sep = "\s*[,]\s*|\s*[:]\s*|[.]",
                        engine = "python")

    names_dict = {}
    #print(names)
    for i in range(names.shape[0]):
        temp = 0
        alist = []
        for j in range(names.shape[1]):
            if temp == 0:
                index = names.iloc[i][j]
                temp = 1
            else:
                if pd.isnull(names.iloc[i][j]): 
                    break
                else:
                    alist.append(names.iloc[i][j])
        names_dict[index] = alist
        temp = 0
                
    col_names = list(names_dict.keys())
    col_names.append("class")
    
    data = pd.read_csv(filenames['data_file'],
                       na_values = '?',
                       header = None,
                       names = col_names)
    #print(data.describe())
    return data, names_dict
 
 
def missing_value_treatment(data, names, key):
    '''
    Assigning  the missing value in each column with the highest value in the
    particular column not already present in  that column.
    '''
    if names[key][0] == 'continuous':
        if data[key].max() > 1:
            mis_val = data[key].max() + 1
            data[key].replace(np.NaN, mis_val, inplace=True)
        else:
            mis_val = data[key].max() + 0.1
            data[key].replace(np.NaN, mis_val, inplace=True)
    else: 
        for i in range(1, len(names[key])+1):
            data[key].replace(names[key][i-1], str(i), inplace=True)
        data[key].replace(np.NaN, len(names[key])+1, inplace=True)
    return


def transform2_numeric(data, names, key):
    '''
    Convert the data into numeric
    '''
    for i in range(1, len(names[key])+1):
        data[key].replace(names[key][i-1], i, inplace=True)
        #print(names[key])


def scaling_dataset(data, names):
    '''
    Min-Max Scaling
    '''
    mm_scaler = MinMaxScaler(feature_range=(0, 10))
    scaler_data = mm_scaler.fit_transform(data)
    scaler_data = pd.DataFrame(scaler_data, columns=list(names.keys()))
    return scaler_data

 
def preprocessing(data, names):
    '''
    Preprocessing the data using the function defined above
    '''
    for key in names.keys():
        if data[key].isnull().any() == True:
            missing_value_treatment(data, names, key)
        if is_string_dtype(data[key]):
            transform2_numeric(data, names, key)
        
    class_data = data['class']
    #print(data.info())
    
    classes = class_data.unique()
    if isinstance(classes[0],str) == False:
        if np.issubdtype(classes[0], np.int64):
            classes.sort()
    
    for i in range(len(classes)):
        class_data.replace(classes[i], i, inplace=True)
        
    #print(data.describe())
    data = data.drop("class", axis=1)
    
    scaled_data = scaling_dataset(data, names)
    return scaled_data, class_data


def distance_plot(data):
    '''
    k-nearest neighbor graph
    '''
    #k = int(round(np.log(len(class_data))))
    
    nearest_neighbors = NearestNeighbors(n_neighbors=10)
    n_neighbor = nearest_neighbors.fit(data)
    distances, indices = n_neighbor.kneighbors(data)
    distances = np.sort(distances, axis=0)[:, 9]
    
    plt.plot(distances)
    plt.grid()
    plt.show() 


def dbscan(data):
    print("DBCSAN......")
    
    db = DBSCAN(eps=3, min_samples=10)
    db.fit(data)
    db_labels = db.labels_
    
    '''
    Displaying number of cluster and noise
    '''    
    core = np.zeros_like(db_labels, dtype=bool)
    core[db.core_sample_indices_] = True
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)
    print("Cluster: %d" % n_clusters)
    print("Noise: %d" % n_noise)
    #print(db_labels)
    return db_labels


def optics(data):
    print("OPTICS....")
    
    db = OPTICS(min_samples=10, xi=0)
    db.fit(data)
    
    '''
    Reachability plot
    '''
    reachability = db.reachability_[db.ordering_]
    db_labels = db.labels_[np.sort(db.ordering_)]
    plt.plot(reachability)
    plt.show()
    
    '''
    Displaying number of cluster and noise
    '''
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)
    print("Cluster: %d" % n_clusters)
    print("Noise: %d" % n_noise)
    
    return db_labels


def cluster_plots(labels, X):
    '''
    PCA converted cluster plot
    '''
    pca = PCA(n_components=2).fit(X)
    pca_2d = pca.transform(X)

    clusters = set(labels)
    pca_2d = pd.DataFrame(pca_2d)
    for cluster in clusters:
        row_ix = np.where(labels == cluster)
        
        if cluster == -1:
            plt.scatter(pca_2d.iloc[row_ix][0], pca_2d.iloc[row_ix][1], c='b', marker='*')
        else:
            plt.scatter(pca_2d.iloc[row_ix][0], pca_2d.iloc[row_ix][1])
    
    plt.show()


def dunn_index(labels, data):
    '''
    Dunn Index
    '''
    '''
    Creating a list of array, where each cluster represent an array  
    '''
    a_list = []
    for i in np.unique(labels):
        class_array = data[labels == i]
        a_list.append(class_array)
    
    '''
    Calculating minimum separation i.e., min(Inter cluster distance) and
    max diameter i.e., max(Intra cluster distance)
    '''
    min_separation = 99999
    max_diameter = 0
    no_of_attr = a_list[0].shape[1]
    for i in range(len(a_list)):
        no_of_rows_i = a_list[i].shape[0]
        for index_i in range(no_of_rows_i):
            for j in range(len(a_list)):
                no_of_rows_j = a_list[j].shape[0]
                for index_j in range(no_of_rows_j):
                    if i == j:
                        distance = 0
                        for k in range(no_of_attr):
                            diff = a_list[i].iloc[index_i, k] - a_list[j].iloc[index_j, k]
                            distance += diff**2
                        distance = math.sqrt(distance)
                        if distance > max_diameter:
                            max_diameter = distance
                    else:
                        distance = 0
                        for k in range(no_of_attr):
                            diff = a_list[i].iloc[index_i, k] - a_list[j].iloc[index_j, k]
                            distance += diff**2
                        distance = math.sqrt(distance)
                        if distance < min_separation:
                            min_separation = distance

    dunn_value = min_separation/max_diameter
    return dunn_value
        
        

def validity_index(dbscan_labels, optics_labels, data, data_labels):
    '''
    #Internal Quality Index
    '''
    
    dbscan_db_index = davies_bouldin_score(data, dbscan_labels)
    optics_db_index = davies_bouldin_score(data, optics_labels)
    dbscan_sil_index = silhouette_score(data, dbscan_labels)
    optics_sil_index = silhouette_score(data, optics_labels)
    dbscan_ch_index = calinski_harabasz_score(data, dbscan_labels)
    optics_ch_index = calinski_harabasz_score(data, optics_labels)
    dbscan_dunn_index = dunn_index(dbscan_labels, processed_data)
    optics_dunn_index = dunn_index(optics_labels, processed_data)
    
    '''
    External Quality Index
    '''
    dbscan_rand = adjusted_rand_score(np.array(data_labels), dbscan_labels)
    optics_rand = adjusted_rand_score(np.array(data_labels), optics_labels)
    dbscan_mutual_info = adjusted_mutual_info_score(np.array(data_labels), dbscan_labels)
    optics_mutual_info = adjusted_mutual_info_score(np.array(data_labels), optics_labels) 
    
    
    print('dbscan...')
    print('db index                - ' + str(dbscan_db_index))
    print('dunn index              - ' + str(dbscan_dunn_index))
    print('silhouette score        - ' + str(dbscan_sil_index))
    print('calinski harabasz score - ' + str(dbscan_ch_index))
    print('Rand index              - ' + str(dbscan_rand))
    print('Mutual Info             - ' + str(dbscan_mutual_info))
    
    print('optics...')
    print('db index                - ' + str(optics_db_index))
    print('dunn index              - ' + str(optics_dunn_index))
    print('silhouette score        - ' + str(optics_sil_index))
    print('calinski harabasz score - ' + str(optics_ch_index))
    print('Rand index              - ' + str(optics_rand))
    print('Mutual Info             - ' + str(optics_mutual_info))
    
    print('DBSCAN contingency matrix')
    print(contingency_matrix(np.array(data_labels), dbscan_labels))
    
    print('OPTICS contingency matrix')
    print(contingency_matrix(np.array(data_labels), optics_labels))
    
    validity_score  = {
                      'DBSCAN DB Index' : dbscan_db_index,
                      'OPTICS DB Index' : optics_db_index,
                      'DBSCAN Dunn Index' : dbscan_dunn_index,
                      'OPTICS Dunn Index' : optics_dunn_index,
                      'DBSCAN Silhouette Score' : dbscan_sil_index,
                      'OPTICS Silhouette Score' : optics_sil_index,
                      'DBSCAN Calinski Harabasz Score' : dbscan_ch_index,
                      'OPTICS Calinski Harabasz Score' : optics_ch_index
                      }
    return validity_score


def write_to_file(score, filename):
    '''
    Writing the Validity Info to a file
    '''
    
    dataname = filename['data_file'].split('.')[0]
    index_data = [score['DBSCAN DB Index'],
                  score['OPTICS DB Index'],
                  score['DBSCAN Dunn Index'],
                  score['OPTICS Dunn Index'],
                  score['DBSCAN Silhouette Score'],
                  score['OPTICS Silhouette Score'],
                  score['DBSCAN Calinski Harabasz Score'],
                  score['OPTICS Calinski Harabasz Score']]
    df = pd.DataFrame([index_data], columns = ['DBSCAN DB Index',
                                               'OPTICS DB Index',
                                               'DBSCAN Dunn Index',
                                               'OPTICS Dunn Index',
                                               'DBSCAN Silhouette Score',
                                               'OPTICS Silhouette Score',
                                               'DBSCAN Calinski Harabasz Score',
                                               'OPTICS Calinski Harabasz Score'], index = [dataname])
    if os.path.exists('cluster_validity.csv'):
        df.to_csv('cluster_validity.csv', mode = 'a', header =False)
    else:
        df.to_csv('cluster_validity.csv')


'''
main body
''' 
 
file = {
            'data_file' : 'iris.data',
            'names_file' : 'iris.names'
            }
(dataDf, namesDict) = read_data(file)
(processed_data, class_data) = preprocessing(dataDf, namesDict)

#processed_data.to_csv('scaled_anneal.data', index=False)

#distance_plot(processed_data)

'''
Density based clustering algorithm
'''

#dbscan_labels = dbscan(processed_data)
#optics_labels = optics(processed_data)

'''
PCA converted cluster plots of following:
1. Original Class label data
2. DBSCAN generated Class label
3. OPTICS generated Class label
'''
#cluster_plots(np.array(class_data), processed_data)
#cluster_plots(dbscan_labels, processed_data)
#cluster_plots(optics_labels, processed_data)

#val_index = validity_index(dbscan_labels, optics_labels, processed_data, class_data)

#write_to_file(val_index, file)

