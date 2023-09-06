from sklearn.preprocessing import StandardScaler
import math


class Knn:
    def __init__(self, K, output, input_data, test_data):
        self.k = K
        self.output = output
        self.input_data = input_data
        self.test_data = test_data

    # normalized data
    def normalized(self, data):
        sc = StandardScaler()
        new = sc.fit_transform(data)
        return new

    # distinguise classes

    # def divide_class(self):
    #     data=self.normalized()
    #     big_list=[]

    #     unique_A=self.output.unique()
    #     for i in range(len(unique_A)):
    #         small_list=[]
    #         for value in range(len(self.output)):
    #             if self.output[value]==unique_A[i]:
    #                 small_list.append(list(data[value]))
    #         big_list.append(small_list)
    #     return big_list

    # calculate_distance
    def calculate_distance(self, point_1, point_2):
        distance = math.sqrt(
            (point_1[0]-point_2[0])**2+(point_1[1]-point_2[1])**2)
        return distance

    # classify one value
    def classify(self, list_data):
        combined = list(zip(list_data, self.output))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        sorted_arr1, sorted_arr2 = zip(*sorted_combined)
        k_sorted=sorted_arr2[:self.k]
        unique_A = self.output.unique()
        check = 0
        for i in unique_A:
            count_ = k_sorted.count(i)
            if count_ > check:
                check = count_
                result = i
        return result

    # classify
    def predict(self):
        data_test = self.normalized(self.test_data)
        train_data = self.normalized(self.input_data)
        list_predicted = []
        for value in data_test:
            list_distances = []
            for y in train_data:
                distance = self.calculate_distance(value, y)
                list_distances.append(distance)
            result = self.classify(list_distances)
            list_predicted.append(result)
        return list_predicted
