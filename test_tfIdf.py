from unittest import TestCase

import pandas as pd

from tf_idf import TfIdf


class TestTfIdf(TestCase):
    def test_cutList(self):
        wordsList = ['don’t get in to a room', 'I have arrived the BeiJing']
        newList=TfIdf().cut(wordsList)
        print(newList)
    def test_get_cuts(self):
        print(TfIdf().getCuts())

    def testAll(self):
        num_clusters = 12  # 分类的个数
        # 以文件的第一行为分割，读取文件，返回一个二维的类似表格
        book_data = pd.read_csv('resource/dataSet1.txt')  # 读取文件

        # book_titles = book_data['title'].tolist()
        book_content = book_data['want to'].tolist()  # 获取content即第书的内容，形成列表,由于是英文他本身就可以作为词袋
        print('--之前book_content--')
        print(book_content)

        newList=TfIdf().cut(book_content)
        print('--去除之后特征 book_content--')
        print(newList)