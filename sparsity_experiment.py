import time
import numpy as np
from torch.serialization import save
import yaml
from experiment import Experiment

class SparsityExperiment(Experiment):

    def __init__(self, data_id, param_set, results_dir):
        super().__init__(data_id, param_set, results_dir)
        
    def run(self):
        tstart = time.perf_counter()
        x, y = self.data.training_data()
        testx, testy = self.data.training_data()
        model = self.model_factory()
        model.show()
        _, sparsity_epochs, loss_epochs = model.learn(x, y)
        pred = model.predict(testx)
        pred_mse = np.sum(np.power((pred - testy), 2)) / pred.size
        tend = time.perf_counter()
        report = {
            "Eval. MSE": pred_mse,
            "Sparsity": model.sparsity(),
            "Sparsity by Epoch": sparsity_epochs,
            "Training Loss by Epoch": loss_epochs,
            "Run Time (S)": tend - tstart
        }
        return lambda: self.save(model, report)


if __name__ == '__main__':
    from data import Data
    from itertools import product
    from deploy import DaskManager, YamlInput
    import sys

    manager = DaskManager(int(sys.argv[1]))
    yamlinput = YamlInput(sys.argv[2])
    results_dir = yamlinput['results_dir']
    pool = []
    
    for data_id, params in yamlinput.iterate_inputs():
        exp = SparsityExperiment(data_id, params, results_dir)
        pool.append(exp.run)
    
    savefns = manager.distributed_run(pool)
    for fn in savefns:
        print(fn())


