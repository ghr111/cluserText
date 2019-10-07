class ClusterData:
    def __init__(self, kmeans_obj, words_data, feature_names, num_clusters, topn_features):
        # cluster_data声明字典，里边存了分类信息
        self.cluster_data = {}
        # kmeans对象，后边需要进行排序
        self.kmeans_obj = kmeans_obj
        self.words_data = words_data
        self.feature_names = feature_names
        self.num_clusters = num_clusters
        self.topn_features = topn_features
        # 执行获取分类信息方法
        self.__format_cluster_data()

    # clustering_obj:keans对象，book_data：书的数据，feature_names：特征名字，num_clusters：要分多少类，topn_features：排序的前几类特征
    def __format_cluster_data(self):
        # 获取cluster的center
        # 对得到的二维数组每个聚类中心点进行排序得到排序索引,这里"::-1"使其按照从大到小排序，ordered_centroids里存的是排序索引
        # 行代表第几类，列代表特征属于这个类的隶属度
        ordered_centroids = self.kmeans_obj.cluster_centers_.argsort()[:, ::-1]

        # 获取每个cluster的关键特征
        for cluster_num in range(self.num_clusters):
            self.cluster_data[cluster_num] = {}
            # 字典的key为cluster_num的值，value为一个字典->key为cluster_num值为cluster_num的值,{1: {'cluster_num': 1, 'key_features': with}}
            self.cluster_data[cluster_num]['cluster_num'] = cluster_num
            # key_features是包含多个features的列表，是for循环，对ordered_centroids二位数组进行遍历，每个类取隶属度最高的topn_features个特征存到key_features中
            key_features = [self.feature_names[index]
                            for index in ordered_centroids[cluster_num, :self.topn_features]]

            self.cluster_data[cluster_num]['key_features'] = key_features
            # 把字典中相同标记的句子归为一类
            same_words = self.words_data[self.words_data['Cluster'] == cluster_num]['I want to'].values.tolist()
            self.cluster_data[cluster_num]['words'] = same_words

    # 显示数据
    def print_cluster_data(self):

        # print cluster details
        for cluster_num, cluster_details in self.cluster_data.items():
            print('Cluster {} details:'.format(cluster_num))
            print('-' * 20)
            print('Key features:', cluster_details['key_features'])
            print('words in this cluster:')
            print('\n'.join(cluster_details['words']))
            print('=' * 40)

    def get_key_features(self):
        keyFeatures = []
        for cluster_num, cluster_details in self.cluster_data.items():
            for feature in cluster_details['key_features']:
                keyFeatures.append(feature)
        return keyFeatures
