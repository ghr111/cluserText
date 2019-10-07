import xlrd


# 词袋 file：传入需要读取的文件
class DataFile:
    def __init__(self, file):
        self.wordsList = []
        self.file = file

    # 读取文件，并且把句子中的单词都提取出来成为列表
    def read_data(self):
        wb = xlrd.open_workbook(filename=self.file)  # 打开文件
        sheet1 = wb.sheet_by_index(0)  # 通过索引获取表格
        for i in range(0, sheet1.nrows):
            cell = sheet1.cell_value(i, 0)
            # rows = sheet1.row_values(i)#获取行内容
            # 转换为字符串，之后可以用split
            str = ''.join(cell)
            # 多个列表里的值放到一个列表中
            self.wordsList = self.wordsList + str.split(' ')
        print("--wordsList:--")
        print(self.wordsList)
        return self.wordsList

    def getRealFeatures(self):
        '''
        从文件中读取评价用到的，正确的特征值
        :return:
        '''
        featureList = []
        with open(self.file, 'r', encoding="utf-8") as f:
            for line in f:
                featureList.append(line.replace("\n",""))#注意去除换行符
        return featureList
