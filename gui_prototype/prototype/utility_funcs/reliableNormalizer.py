"""
@file: reliableNormalizer.py
Created on 15.01.2017 15:53
@project: GitHubRepositoryClassifier

@author: Lukas

don't use this, apparently I'm stupid
"""

import math

class ReliableNormalizer:

    def __init__(self, use_log=True):
        self.use_log = use_log
        self.lst_values = None


    def log(self, input_array):
        print(input_array)
        # if isinstance(input_array[0], list):
        input_array_tmp = [None] * len(input_array[0])
        input_array_tmp_big = []
        for z in range(0, len(input_array)):
            for i in range(0, len(input_array[z])):
                if input_array[z][i] > 0:
                    # print(input_array[z][i])
                    input_array_tmp[i] = math.log2(input_array[z][i] + 1)

                else:
                    input_array_tmp[i] = 0
            input_array_tmp_big.append(input_array_tmp[:])

        #elif isinstance(input_array, list):
         #   input_array_tmp_big = []
          #  for i in range(0, len(input_array)):
           #     if input_array[i] > 0:
            #        # print(input_array[z][i])
             #       input_array_tmp_big[i] = math.log2(input_array[i] + 1)

        #        else:
         #           input_array_tmp_big[i] = 0
        #else:
         #   raise Exception("Error, input is not a handled Datatype")

        return input_array_tmp_big



    # calculate parameter
    def fit(self, input_array):
        print("input: " + str(input_array))
        lst_temp = [0] * len(input_array[0])

        if self.use_log:
            input_array_tmp_big = self.log(input_array)
        else:
            input_array_tmp_big = input_array[:]

        for values in input_array_tmp_big:
            # print(values)
            for i in range(0, len(values)):
                lst_temp[i] += values[i] * values[i]

        print("total: " + str(lst_temp))

        lst_result = [0] * len(lst_temp)
        for i in range(0, len(lst_temp)):
            lst_result[i] = math.sqrt(lst_temp[i])

        print("result: " + str(lst_result))

        self.lst_values = lst_result[:]

        return self

    # Object is transformed
    def transform(self, input_array):

        lst_result = []

        if self.use_log:
            input_array_tmp_big = self.log(input_array)
        else:
            input_array_tmp_big = input_array[:]

        input_array_tmp = [None] * len(input_array_tmp_big[0])

        for i in range(0, len(input_array_tmp_big)):
            for j in range(0, len(input_array_tmp_big[i])):
                if self.lst_values[j] != 0:
                    input_array_tmp[j] = input_array_tmp_big[i][j] / self.lst_values[j]
                else:
                    input_array_tmp[j] = 0
            lst_result.append(input_array_tmp[:])
            # print(input_array_tmp)

        return lst_result

    def fit_transform(self, input_array):
        self.fit(input_array)
        return self.transform(input_array)

