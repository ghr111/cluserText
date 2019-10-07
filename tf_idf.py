from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# tf_idf算法输入词袋，进行归一化标准化成特征矩阵
class TfIdf:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    # 获取剪词列表，里边放的是介词，里边出现的词不能作为特征值
    def getCuts(self):
        with open('resource/cut_words.utf8') as file:
            cutList = []
            for line in file.readlines():
                cutList.append(line.strip('\n'))
        return cutList
    # 进行剪词
    def cut(self, wordslist):
        cutList=self.getCuts()
        new_word_list = []
        # 遍历词袋里的语句
        for str in wordslist:
            # 把一句话分成一个数组
            # print(str)
            # print('*' * 10)
            array = str.split(' ')

            low_set=set()
            for item in array:
                if '’' in item:# 去除带有缩写的单词
                    continue
                low_set.add(item.lower())
            for cut in cutList:
                if cut in low_set:
                    low_set.remove(cut)
            # 以空格为分隔符生成字符串
            new_str = ' '.join(low_set)
            # 生成新的词袋，里边不包含剪词列表的所有单词
            new_word_list.append(new_str)
        return new_word_list
    # 建立特征矩阵
    def doItfIdf(self, wordsList):
        # 进行剪词，去除介词
        cutedWordList=self.cut(wordsList)
        print('--cuttedWordList--')
        print(cutedWordList)
        # fit_transform(x)	x是列表，每个列表包含一行文本，返回一个字典，key：词，value:频(？这里的词频为什么与输入的元素无关)
        # feature_matrix即tf_idf结果，fit_transform http://blog.sina.com.cn/s/blog_b8effd230102yznw.html
        # 数据归一化、标准化 https://www.cnblogs.com/pejsidney/p/8031250.html
        feature_matrix = self.vectorizer.fit_transform(cutedWordList).astype(float)
        # print("显示特征矩阵")
        # print(feature_matrix)
        # python返回多参数 https://www.cnblogs.com/wllhq/p/8119347.html
        return self.vectorizer, feature_matrix