from experiment import Experiment
import numpy as np
import time

class Experiment1(Experiment):

    def __init__(self, data, param_set):
        super().__init__(data, param_set, results_dir="experiment_1_server")

    def run(self):
        tstart = time.perf_counter()
        x, y = self.data.training_data()
        print(y[:5])
        print("==")
        testx, testy = self.data.training_data()
        model = self.model_factory()
        model.show()
        _, sparsity_epochs, loss_epochs = model.learn(x, y)
        pred = model.predict(testx)
        pred_mse = np.round(np.sum(np.power((pred - testy), 2)) / pred.size, 2)
        pred_acc = np.round((np.sign(pred) == np.sign(testy)).sum() / pred.size, 2)
        tend = time.perf_counter()
        report = {
            "Eval. MSE": pred_mse,
            "Eval. Acc": pred_acc,
            "Sparsity": model.sparsity(),
            "Sparsity by Epoch": sparsity_epochs,
            "Training Loss by Epoch": loss_epochs,
            "Run Time (S)": tend - tstart
        }
        return lambda: self.save(self.data, model, report)

if __name__ == '__main__':
    from data import Data
    from itertools import product
    from dask_manager import Manager
    import sys

    manager = Manager(int(sys.argv[1]))

    data = Data(4)
    data.save(override=False)
    data.load()

    epochs = [50000]
    relu_widths = [data.D*data.D*data.n]
    linear_widths = [data.D*data.D]  # , data.D*data.D*data.n]
    lambdas = [0.001, 0.01, 0.1]
    terms = [1, 2]


    pool = []
    for rw, lw, e, lam, term in product(relu_widths, linear_widths, epochs, lambdas, terms):
        params = {"relu_width": rw, "linear_width": lw, 
        "layers": 3, "epochs": e, "learning_rate": 1e-3,
        "regularization_lambda": lam, "regularization_method": term}
        print(params)
        exp = Experiment1(data, params)
        pool.append(exp.run)

    savefns = manager.distributed_run(pool)
    for fn in savefns:
        print(fn())