from sklearn.cluster import KMeans


# kmeans算法可以把特征矩阵进行分类
class KMeansExe:
    def __init__(self, feature_matrix, num_clusters=10):
        self.feature_matrix = feature_matrix
        self.kmeans = KMeans(n_clusters=num_clusters, max_iter=10000)

    # k_means:https://blog.csdn.net/lynn_001/article/details/86679270
    def doKmeans(self):
        # k_means开始拟合分类
        self.kmeans.fit(self.feature_matrix)
        # cluster存已经分好类的标签,是数字标明的列表，相同的数字代表是一类，和输入的wordList顺序一一对应
        clusters = self.kmeans.labels_
        return self.kmeans, clusters
