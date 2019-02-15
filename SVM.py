import random
import numpy as np
import time


def load_data(filename):
    feature = []
    label = []
    f = open(filename)
    for line in f.readlines():
        line_data = line.split()
        arr = []
        for i in range(0, len(line_data) - 1):
            arr.append(float(line_data[i]))
        feature.append(arr)
        label.append(float(line_data[-1]))
    data_nums = len(feature)
    return feature, label, data_nums

def load_test_data(filename):
    feature = []
    f = open(filename)
    for line in f.readlines():
        line_data = line.split()
        arr = []
        for i in range(0, len(line_data)):
            arr.append(float(line_data[i]))
        feature.append(arr)
    data_nums = len(feature)
    return feature, data_nums


class data_info:
    def __init__(self, feature, label, constance, t, method):
        self.get_intercept()
        self.data_matrix = feature  # data feature
        self.get_tolerance(t)
        self.get_shape(feature)
        self.get_initial_alphas()
        self.get_record_cache()
        self.get_initial_kernel()
        self.get_kernel(method)
        self.label_matrix = label  # data label
        self.C = constance  # constant C related to slack variable

    def get_tolerance(self, t):
        self.toler = t

    def get_intercept(self):
        self.b = 0

    def get_initial_kernel(self):
        self.K = np.matrix(np.zeros((self.m, self.m)))

    def get_record_cache(self):
        self.e_cache = np.matrix(np.zeros((self.m, 2)))  # sign bit and real error

    def get_initial_alphas(self):
        self.alphas = np.matrix(np.zeros((self.m, 1)))

    def get_shape(self, feature):
        self.m = np.shape(feature)[0]  # the length of the feature

    def get_kernel(self, method):
        for i in range(self.m):  # the global kernel value
            self.K[:, i] = kernel(self.data_matrix, self.data_matrix[i, :], method)


#
# def select_random(index, m):
#     result = index
#     while result == index:
#         result = random.randint(0, m - 1)
#     return result

def calculate_error(data_info, j):
    # calculate the error between current_fx and its label
    alphas_y = np.multiply(data_info.alphas, data_info.label_matrix).transpose()  # result of alphas multiplies y
    kernel_x = data_info.K[:, j]
    b = data_info.b
    current_fx = float(alphas_y * kernel_x + b)
    error = current_fx - float(data_info.label_matrix[j])
    return error


def kernel(matrix, A, method):  # kernel(self.data_matrix, self.data_matrix[i, :], method)
    m, temp = prepare(matrix)
    if method[0] == 'Linear_kernel':
        temp = Linear_kernel(A, matrix)
    elif method[0] == 'Gaussian_kernel':
        temp = Guassian_kernel(A, m, matrix, method, temp)
    else:
        print("error")
    return temp


def prepare(matrix):
    row_num, _ = np.shape(matrix)
    temp = np.matrix(np.zeros((row_num, 1)))
    return row_num, temp


def Guassian_kernel(row, m, matrix, method, temp):
    for j in range(m):
        k = matrix[j, :] - row  # the row of matrix minus matrix
        temp[j] = k * k.transpose()  # ||x-y||^2
    # temp = np.exp(temp / (-1 * method[1] ** 2))
    reach_rate = method[1]
    denominator = reach_rate * reach_rate * (-1)
    inner_value = temp / denominator
    temp = np.exp(inner_value)
    return temp


def Linear_kernel(A, matrix):
    temp = matrix * A.transpose()
    return temp


def heuristic_select(i, data_info, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    data_info.e_cache[i] = [1, Ei]
    valiEcacheList = np.nonzero(data_info.e_cache[:, 0].A)[0]  # choose no-zero index
    if len(valiEcacheList) > 1:
        return choose_step(Ei, Ej, data_info, i, maxDeltaE, maxK, valiEcacheList)
    else:
        # j = select_random(i, data_info.m)
        j = i
        while j == i:
            j = random.randint(0, data_info.m - 1)
        Ej = calculate_error(data_info, j)
    return j, Ej


def choose_step(Ei, Ej, data_info, i, maxDeltaE, maxK, no_zero_index):  # choose the max step
    for k in no_zero_index:
        if k == i:
            continue
        Ek = calculate_error(data_info, k)
        deltaE = abs(Ei - Ek)
        if deltaE > maxDeltaE:
            maxK = k
            maxDeltaE = deltaE
            Ej = Ek
        else:
            pass
    return maxK, Ej


def updateEk(data_info, k):
    Ek = calculate_error(data_info, k)
    # alphas_y = np.multiply(data_info.alphas, data_info.label_matrix).transpose()  # result of alphas multiplies y
    # kernel_x = data_info.K[:, j]
    # b = data_info.b
    # current_fx = float(alphas_y * kernel_x + b)
    # error = current_fx - float(data_info.label_matrix[j])
    data_info.e_cache[k] = [1, Ek]


def adjust_alpha(a, left, right):  # a is alphas values and to keep  left > a > right
    if a > left:
        a = left
    else:
        pass
    if a < right:
        a = right
    else:
        pass
    return a


def inner_Algorithm(i, data):  # data.label_matrix[i] * Ei > data.toler
    Ei = calculate_error(data, i)
    if ((data.alphas[i] < data.C) and (data.label_matrix[i] * Ei < -data.toler)) or (
            (data.alphas[i] > 0) and (data.label_matrix[i] * Ei > data.toler)):
        Ej, alphaIold, alphaJold, j = choose_record(Ei, data, i)
        high, low = up_lower_bound(data, i, j)
        if high == low:
            return 0
        eta = calculate_eta(data, i, j)
        if eta >= 0:
            return 0
        update_alphas_j(Ei, Ej, data, eta, high, j, low)
        if abs(data.alphas[j] - alphaJold) < 0.00001:
            return 0
        update_alpha_i(alphaJold, data, i, j)
        b1, b2 = update_intercept(Ei, Ej, alphaIold, alphaJold, data, i, j)
        calculate_intercept(b1, b2, data, i, j)
        return 1
    else:
        return 0


def calculate_intercept(b1, b2, data, i, j):
    if (0 < data.alphas[i]) and (data.C > data.alphas[i]):
        data.b = b1
    elif (0 < data.alphas[j]) and (data.C > data.alphas[j]):
        data.b = b2
    else:
        data.b = (b1 + b2) / 2.0


def update_intercept(Ei, Ej, alphaIold, alphaJold, data, i, j):
    b1 = calculate_b1(Ei, alphaIold, alphaJold, data, i, j)
    b2 = calculate_b2(Ej, alphaIold, alphaJold, data, i, j)
    return b1, b2


def calculate_b2(Ej, alphaIold, alphaJold, data, i, j):
    return data.b - Ej - data.label_matrix[i] * (data.alphas[i] - alphaIold) * data.K[i, j] - data.label_matrix[j] * (
            data.alphas[j] - alphaJold) * data.K[j, j]


def calculate_b1(Ei, alphaIold, alphaJold, data, i, j):
    b1 = data.b - Ei - data.label_matrix[i] * (data.alphas[i] - alphaIold) * data.K[i, i] - data.label_matrix[j] * (
            data.alphas[j] - alphaJold) * data.K[i, j]
    return b1


def update_alpha_i(alphaJold, data, i, j):
    data.alphas[i] += data.label_matrix[j] * data.label_matrix[i] * (alphaJold - data.alphas[j])
    updateEk(data, i)


def update_alphas_j(Ei, Ej, data, eta, high, j, low):
    data.alphas[j] -= data.label_matrix[j] * (Ei - Ej) / eta
    data.alphas[j] = adjust_alpha(data.alphas[j], high, low)
    updateEk(data, j)


def calculate_eta(data, i, j):
    eta = 2.0 * data.K[i, j] - data.K[j, j] - data.K[i, i]
    return eta


def up_lower_bound(data, i, j):
    if (data.label_matrix[i] != data.label_matrix[j]):
        low = max(0, data.alphas[j] - data.alphas[i])
        high = min(data.C, data.C + data.alphas[j] - data.alphas[i])
    else:
        low = max(0, data.alphas[j] + data.alphas[i] - data.C)
        high = min(data.C, data.alphas[j] + data.alphas[i])
    return high, low


def choose_record(Ei, data, i):  # choose the max step and record old value
    j, Ej = heuristic_select(i, data, Ei)
    previous_alphaI = data.alphas[i].copy()
    previous_alphaJ = data.alphas[j].copy()
    return Ej, previous_alphaI, previous_alphaJ, j


def traverse_nobound(C, changed, data, iter):
    nonbounds = np.nonzero((data.alphas.A > 0) * (data.alphas.A < C))[0]
    for i in nonbounds:
        changed += inner_Algorithm(i, data)
        # print("non_bound_iter: ", iter, "i:", i, "pairchanged: ", alphaPairsChanged)
    iter += 1
    return changed, iter


def traverse_all(changed, data, iter, m):
    for i in range(m):
        changed += inner_Algorithm(i, data)
        # print("full_set_iter: ", iter, "i:", i, "pairchanged: ", alphaPairsChanged)
    iter += 1
    return changed, iter


def prepare_data(C, feature, label, method, toler):
    data_matrix = np.matrix(feature)
    label_matrix = np.matrix(label).transpose()
    m, n = np.shape(data_matrix)
    # alphas = np.matrix(np.zeros((m, 1)))  # m rows and 1 column
    data = data_info(data_matrix, label_matrix, C, toler, method)  # create a data object
    count = 0
    flag = True
    changed = 0
    return changed, data, flag, count, m


def platt_SMO(train_feature, label, C, toler, maxIter, time_budget, method=('Linear_kernel', 0)):
    changed, data, flag, count, m = prepare_data(C, train_feature, label, method, toler)
    start_time = time.time()
    while count < maxIter and changed > 0 or flag:
        changed = 0
        if flag:
            changed, count = traverse_all(changed, data, count, m)
        else:
            changed, count = traverse_nobound(C, changed, data, count)
        if flag:
            flag = False
        elif changed == 0:
            flag = True
    return data.b, data.alphas


# 梯度下降法--------------------------------------------------------------------------------------------------------------
def Gaussian_kernel_train(time_budget, feature, label):
    time1 = time.time()
    parameter = 150
    train_feature = feature[0:int(len(feature) * 0.9)]
    train_label = label[0:int(len(label) * 0.9)]
    # print(len(train_feature), len(train_label))
    time_budget = time_budget - (time.time() - time1)
    b, alphas = platt_SMO(train_feature, train_label, 15, 0.0001, 1000, time_budget, ('Gaussian_kernel', parameter))
    data_matrix = np.matrix(train_feature)
    label_matrix = np.matrix(train_label).transpose()
    # 训练在这里写
    index = np.nonzero(alphas.A > 0)[0]
    support_vector = data_matrix[index]
    label_vector = label_matrix[index]
    # print("execute time", time.time() - time1)
    return support_vector, label_vector, alphas, index, b


def test(support_vector, label_vector, alphas, index, b, test_feature):
    parameter = 150
    data_matrix_ = np.matrix(test_feature)
    m_, n_ = np.shape(data_matrix_)
    for i in range(m_):
        kernelEval = kernel(support_vector, data_matrix_[i, :], ('Gaussian_kernel', parameter))
        predict = kernelEval.T * np.multiply(label_vector, alphas[index]) + b
        print(int(np.sign(predict)))
    # print("test_error_rate: ", float(error_count / m_) * 100, "%")


class Gradient:
    def __init__(self, x, y):
        self.x = np.c_[np.ones((x.shape[0])), x]
        self.y = y
        self.get_epochs()
        self.set_learning_rate()
        self.get_random_w()

    def get_random_w(self):
        self.w = np.random.uniform(size=np.shape(self.x)[1], )

    def set_learning_rate(self):
        self.learning_rate = 0.01

    def get_epochs(self):
        self.epochs = 200

    def get_loss(self, x, y):
        loss = max(0, 1 - y * np.dot(x, self.w))
        return loss

    def cal_sgd(self, x, y, wi):
        if y * np.dot(x, wi) < 1:
            wi = self.calculate_w(wi, x, y)
        else:
            wi = wi
        return wi

    def calculate_w(self, wi, x, y):
        wi = wi - self.learning_rate * (-y * x)
        return wi

    def train(self):
        for epoch in range(self.epochs):
            randomsize = np.arange(len(self.y))
            np.random.shuffle(randomsize)
            x = np.array(self.x)[randomsize]
            y = np.array(self.y)[randomsize]
            loss = 0
            for xi, yi in zip(x, y):
                loss += self.get_loss(xi, yi)
                self.w = self.cal_sgd(xi, yi, self.w)
            # print("epoch:", self.epochs, "loss:", loss)

    def predict(self, x1, y1, x):
        x_train = np.c_[np.ones((x1.shape[0])), x1]
        predict_train = np.sign(np.dot(x_train, self.w))

        x_test = np.c_[np.ones((x.shape[0])), x]
        predict = np.sign(np.dot(x_test, self.w))

        error_rate = 0
        m = np.shape(predict_train)[1]
        result_train = predict_train.tolist()[0]
        label = y1.tolist()[0]
        # print(len(result))
        # print(len(label))
        for i in range(m):
            if result_train[i] != label[i]:
                error_rate += 1
        # print(1-error_rate/m)
        result = predict.tolist()[0]
        return result, 1 - error_rate / m   # 训练结果，预测结果，准确率


# -------------------------------Test area-------------------------------
# feature, label = load_data("testSet.txt")
# # b, alphas = simple_SMO(feature, label, 0.6, 0.001, 40)
# w = get_w(feature, label, alphas)
# showClassifer(feature, w, b)
# print(b, "    ", alphas)
# feature, label = load_data("testSet.txt")
# b, alphas = platt_SMO(feature, label, 0.6, 0.001, 40)
# w = calcWs(alphas, feature, label)
# showClassifer(feature, label, w, b)
import sys

if __name__ == "__main__":
    start = time.time()
    time_buget = 60
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    # training-------------------------------------------------------
    feature, label, z = load_data(train_file)
    matrix = np.matrix(feature)
    label_ = np.matrix(label)
    data = Gradient(matrix, label)
    data.train()
    # ---------------------------------------------------------------
    # testing--------------------------------------------------------
    test_feature,  m_test = load_test_data(test_file)
    test_matrix = np.matrix(test_feature)
    result, accuracy = data.predict(matrix, label_, test_matrix)
    if accuracy > 0.7:
        for i in range(m_test):
            print(int(result[i]))
        # print(accuracy)
    else:
        support_vector, label_vector, alphas, index, b = Gaussian_kernel_train(time_buget, feature, label)
        test(support_vector, label_vector, alphas, index, b, test_feature)
    # print("total time: ", time.time() - start)
    # error_max = 1
    # record = defaultdict(float)
    # optimal = defaultdict(float)
    # for i in range(120):
    #     random.seed(i)
    #     print("The seed is: ", i)
    #     error_rate = Gaussian_kernel_train(time_buget)[0]
    #     print("error_rate: ", error_rate)
    #     record[i] = error_rate
    #     if error_rate < error_max:
    #         error_max = error_rate
    #         optimal[i] = error_rate
    # print(record)
    # print(" ")
    # print(optimal)
