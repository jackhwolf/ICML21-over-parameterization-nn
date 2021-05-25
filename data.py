import numpy as np
import os
from itertools import product

class Data:
    ''' generate a random sample of (feature,label) pairs '''

    def __init__(self, D=2, N=1000, n=32, preloaded=False):
        '''
        D: int, feature dimensions
        N: int, total # points  (D<=2)
        n: int, # points to randomly samply for training (D<=2)
        '''
        self.D = int(D)
        self.N = int(N)
        self.n = int(n)
        self.X = self.build_x()
        self.Y = self.build_y()
        self.mask = np.array([False]*self.N)
        self.mask[:self.n] = True
        np.random.shuffle(self.mask)
        if preloaded:
            self.load()

    def save(self, override=False):
        os.makedirs("datasets", exist_ok=True)
        fname = self._fname()
        data = [self.X, self.Y, self.mask]
        exts = ["X.npy", "Y.npy", "mask.npy"]
        if os.path.exists(f"datasets/{fname}_{exts[0]}") and not override:
            return False
        for i in range(len(data)):
            path = f"datasets/{fname}_{exts[i]}"
            np.save(path, data[i])
        return True

    def load(self):
        fname = self._fname()
        if not os.path.exists(f"datasets/{fname}_X.npy"):
            return False
        self.X = np.load(f"datasets/{fname}_X.npy")
        self.Y = np.load(f"datasets/{fname}_Y.npy")
        self.mask = np.load(f"datasets/{fname}_mask.npy")
        return True

    def _fname(self):
        fname = f"D={self.D}"
        if self.D <= 2:
            fname += f"_N={self.N}_n={self.n}"
        return fname

    def build_x(self):
        if self.D in [1, 2]:
            return np.random.uniform(low=-1, high=1, size=(self.N,self.D))
        elif self.D in [3, 4]:
            x = [0, 0.5, 1]
        x = list(product(x, repeat=self.D))
        self.N = len(x)
        self.n = len(x)
        return np.array(x)

    def build_y(self):
        y = []
        if self.D == 1:
            y = self.spline(self.X)
        elif self.D == 2:
            y = self.sin_2d(self.X)
        else:
            y = self.rvf(self.X)
        return y.reshape(-1,1)

    def describe(self):
        return {"D": self.D, "N": self.N, "n": self.n}

    def training_data(self):
        if self.D <= 2:
            return (self.X[self.mask,:], self.Y[self.mask])
        return (self.X, self.Y)

    def testing_data(self):
        if self.D <= 2:
            return (self.X[~self.mask,:], self.Y[~self.mask])
        return (self.X, self.Y)

    @staticmethod
    def spline(x):
        return (np.abs(x) >= 0.5).astype(int)*2 - 1

    @staticmethod
    def sin_2d(x):
        return np.sign(x[:,1] - (0.5*np.sin(np.pi*x[:,0])))

    @staticmethod
    def rvf(x):
        return 2*np.random.rand(x.shape[0]) - 1


def graph_1d_sample():
    import matplotlib.pyplot as plt
    import os
    os.makedirs("./sample_data", exist_ok=True)
    d = Data(1, 1000, 1)
    x, y = d.X, d.Y
    fig, ax = plt.subplots(figsize=(8,4))
    fig.suptitle("Sample 1D Data", fontsize=15)
    cbar = ax.scatter(x, y, c=y, cmap="bwr")
    fig.colorbar(cbar)
    fig.savefig("./sample_data/sample_data_1d.png", facecolor="white", bbox_inches="tight")

def graph_2d_sample():
    import matplotlib.pyplot as plt
    import os
    os.makedirs("./sample_data", exist_ok=True)
    n = 500
    fig, ax = plt.subplots(figsize=(10,6), subplot_kw={"projection": "3d"})
    x = np.linspace(-1, 1, n)
    x, y = np.meshgrid(x, x)
    x, y = x.flatten(), y.flatten()
    z = []
    for i in range(len(x)):
        z.append(Data.sin_2d(np.array([[x[i], y[i]]])))
    x = x.reshape(n,n)
    y = y.reshape(n,n)
    z = np.array(z).reshape(n,n)
    cbar = ax.plot_surface(x, y, z, cmap='bwr', linewidth=0, antialiased=False)
    fig.colorbar(cbar, shrink=0.5, aspect=5)
    fig.suptitle("y = sign(x[1] - 0.5*sin(pi*x))", fontsize=16)
    fig.savefig("./sample_data/sample_data_2d.png", bbox_inches='tight')

if __name__ == "__main__":
    rvf = Data(4)
    print(rvf.Y[:5])
    rvf.save()
    # rvf.save()

    rvf2 = Data(4)
    print(rvf2.Y[:5])
    rvf2.load()
    print(rvf2.Y[:5])
