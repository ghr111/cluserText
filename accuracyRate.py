class AccuracyRate:
    def __init__(self, realFeatures,forecastFeatures):
        self.realFeatures=realFeatures
        self.forecastFeatures=forecastFeatures
    def getAccuracyRate(self):
        sum=0#记录所有的属性个数

        counter=0# 记录分类正确的属性
        for forecastFeature in self.forecastFeatures:
            sum = sum + 1
            for realFeature in self.realFeatures:
                # print(forecastFeature)

                if  realFeature in forecastFeature:
                    counter=counter+1
        return (counter/sum)*100