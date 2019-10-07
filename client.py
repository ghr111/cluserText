import time

from accuracyRate import AccuracyRate
from clusterData import ClusterData
from dataFile import DataFile
from kmeans import KMeansExe
from tf_idf import TfIdf
import pandas as pd

# 每次聚类结果不一样，这是因为kmeans本身存在的缺陷：初始化的点每次都是随机的
def doCluster(file_name,num_clusters):
    num_clusters = num_clusters  # 分类的个数
    topn_features = 1  # 分类选取的特征值数
    # 以文件的第一行为分割，读取文件，返回一个二维的类似表格
    words_data = pd.read_csv('resource/'+file_name+'.txt')  # 读取文件
    # print(book_data)

    # book_titles = book_data['title'].tolist()
    words_content = words_data['I want to'].tolist()  # 获取content即第书的内容，形成列表,由于是英文他本身就可以作为词袋
    # print(words_content)
    # print('--book_content--')
    # print(book_content)

    # tf-idf 计算特征矩阵
    # 获取TfidfVectorizer向量化构造器，获取特征矩阵
    vectorizer, feature_matrix = TfIdf().doItfIdf(words_content)
    # # 查看特征矩阵
    # print('--查看特征矩阵规模 feature_matrix.shape--')
    # print(feature_matrix.shape)

    # 获取特征名字
    feature_names = vectorizer.get_feature_names()
    # 打印某些特征,显示前10个
    # print('--打印了特征 feature_names[:]--')
    # print(feature_names[:])

    # 放入词袋，放入需要分类的数量，运行keans方法进行聚类，获取KMeans对象，获取已经分好类的标签列表clusters
    kmeans_obj, clusters = KMeansExe(feature_matrix, num_clusters).doKmeans()

    # 存储每个单词对应的分类标记
    words_data['Cluster'] = clusters.tolist()
    # print('--book_data--')
    # print(book_data)
    # 获取显示分好类的数据
    clusterdata = ClusterData(kmeans_obj, words_data=words_data, feature_names=feature_names, num_clusters=num_clusters,
                              topn_features=topn_features)
    # 显示详细分类
    clusterdata.print_cluster_data()

    # 计算正确率
    realFeatures=DataFile("resource/realFeatures_"+file_name).getRealFeatures()
    # print('-'*7)
    # print(realFeatures)
    # realFeatures = ['deposit', 'maintain', 'dataset', 'manage', 'access', 'protect', 'guarantee', 'track', 'link',
    #                 'search', 'view', 'see', 'get', 'harvest', 'record', 'archiv']
    forecastFeatures = clusterdata.get_key_features()
    print(forecastFeatures)
    accuracyRate = AccuracyRate(realFeatures, forecastFeatures).getAccuracyRate()
    print(str(accuracyRate) + "%")

doCluster('dataSet5', 10)
# if __name__ == '__main__':
#     for i in range(0,5):
#         doCluster('dataSet5', 16)
#         time.sleep(1)
