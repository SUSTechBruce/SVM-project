import numpy as np



class Gradient:
    def __init__(self, x, y, epochs=200, learning_rate=0.01):
        self.x = np.c_[np.ones((x.shape[0])), x]
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = np.random.uniform(size=np.shape(self.x)[1], )

    def get_loss(self, x, y):
        loss = max(0, 1 - y * np.dot(x, self.w))
        return loss

    def cal_sgd(self, x, y, w):
        if y * np.dot(x, w) < 1:
            w = w - self.learning_rate * (-y * x)
        else:
            w = w
        return w

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
            print("epoch:", self.epochs, "loss:", loss)
    def predict(self, x, y):
        x_test = np.c_[np.ones((x.shape[0])), x]
        predict = np.sign(np.dot(x_test, self.w))
        error_rate = 0
        m = np.shape(predict)[1]
        result = predict.tolist()[0]
        label = y.tolist()[0]
        print(len(result))
        print(len(label))
        for i in range(m):
            if result[i] != label[i]:
                error_rate += 1
        print(1-error_rate/m)
        return np.sign(np.dot(x_test, self.w)), 1-error_rate/m

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
    return feature, label
import time
if __name__ == "__main__":
    start = time.time()
    feature, label = load_data("train_data.txt")
    matrix = np.matrix(feature)
    label_ = np.matrix(label)
    data = Gradient(matrix, label)
    data.train()
    print(data.predict(matrix,label_))
    print(time.time()-start)