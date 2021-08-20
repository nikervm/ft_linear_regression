import sys
import regression
import pandas as pd
import numpy as np
from sklearn import datasets
import random


def test_program(n, type):
    try:
        n = int(n)
    except Exception as error:
        print(error)
        exit(0)
    if type == "1":
        x = np.arange(-n / 2, n / 2, 1, dtype=np.float64)

        m = np.random.uniform(0.3, 0.5, (n,))
        b = np.random.uniform(5, 10, (n,))

        y = x * m + b
    else:
        if random.randint(0, 9) % 2 == 0:
            c = -1
        else:
            c = 1
        x, y = datasets.make_regression(n_samples=n,
                                        n_features=1,
                                        noise=50)
        x = np.interp(x, (x.min(), x.max()), (-240000, 240000))
        x = [x[0] for x in x]
        y = np.interp(y, (y.min(), y.max()), (-c * 20, c * 20))
    model.train(np.array(x), np.array(y))
    model.plot_data()


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2 or len(args) > 4:
        exit(0)
    model = regression.linear_regression()
    if len(args) == 4 and args[1] == "--test" and (args[3] == "1" or args[3] == "2"):
        test_program(args[2], args[3])
        exit(0)
    if len(args) != 2:
        exit(0)
    try:
        df = pd.read_csv(sys.argv[1])
        model.train(df['km'].to_numpy(), df['price'].to_numpy())
        model.plot_data()
    except Exception as e:
        print(e)
