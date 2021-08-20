import numpy as np
from matplotlib import pyplot as plt


class linear_regression:
    def __init__(self, learning_rate=1.0):
        self.delta_error = 10 ** -6
        self.learning_rate = learning_rate
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.prev_error = 0.0
        self.cur_error = 0.0
        self.error = 10 ** -6
        self.step_exit = 10 ** 3
        self.X = []
        self.y = []

    def __call__(self, y):
        return self.predict(y)

    def __normilize(self):
        return (self.X - np.min(self.X)) / (np.max(self.X) - np.min(self.X)),\
               (self.y - np.min(self.y)) / (np.max(self.y) - np.min(self.y))

    def train(self, X, y):
        self.X = X
        self.y = y
        trainX, trainY = self.__normilize()
        delta = 100
        step = 1
        while abs(delta) > self.error or step > self.step_exit:
            y_pred = self.predict(trainX)
            self.theta0 -= self.learning_rate * (y_pred - trainY).mean()
            self.theta1 -= self.learning_rate * ((y_pred - trainY) * trainX).mean()
            self.prev_error = self.cur_error
            self.cur_error = np.sum(np.power(y_pred - y, 2))
            delta = self.cur_error - self.prev_error
            step += 1
        self.theta1 = (np.max(y) - np.min(y)) * self.theta1 / (np.max(X) - np.min(X))
        self.theta0 = np.min(y) + ((np.max(y) - np.min(y)) * self.theta0) + self.theta1 * (1 - np.min(X))
        self.__save()

    def predict(self, X):
        return self.theta0 + self.theta1 * X

    def __save(self):
        np.savetxt("indexes.csv", np.array([self.theta0, self.theta1]))

    def load(self):
        self.theta0, self.theta1 = np.loadtxt("indexes.csv")

    def plot_data(self):
        plt.title("Linear regression")
        plt.xlabel("Mileage")
        plt.ylabel("Price")
        pred = self.predict(self.X)
        plt.plot(self.X, self.y, 'ro', self.X, pred)
        plt.show()
