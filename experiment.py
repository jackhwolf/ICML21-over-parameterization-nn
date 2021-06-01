import matplotlib
from torch.serialization import save
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Model
import numpy as np
import os
from uuid import uuid4
import json
from torch import save as torchsave
from scipy.io import savemat
import time

class Experiment:

    def __init__(self, data, param_set, results_dir="final/all_terms"):
        results_dir = "experiments/" + results_dir
        self.data = data
        self.model_factory = lambda: Model(data.D, **param_set)
        self.results_dir = results_dir
        self.result_path = f"{self.results_dir}/results.json"
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self):
        tstart = time.perf_counter()
        x, y = self.data.training_data()
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

    def save(self, data, model, report):
        out = {
            'data': self.data.describe(),
            'model': model.describe(),
            'report': report
        }
        out_c = out.copy()
        plt.close('all')
        if self.data.D == 1:
            out['interpolation_sparsity_path'] = self.graph_1d_interpolation(data, model, out_c)
        elif self.data.D == 2:
            out['interpolation_sparsity_path'] = self.graph_2d_interpolation(data, model, out_c)
        else:
            out['interpolation_sparsity_path'] = self.graph_Nd_learned_sparsity(data, model, out_c)
        out['sparsity_by_epoch_path'] = self.graph_sparsity_by_epoch(data, model, out_c)
        out['state_dict_path'] = self.save_state_dict(model)
        out['matlab_path'] = self.save_state_dict_matlab(model)
        if not os.path.exists(self.result_path):
            open(self.result_path, 'a').close()
        curr = None
        with open(self.result_path, 'r') as fp:
            curr = fp.read()
            if curr:
                curr = json.loads(curr)
            else:
                curr = []
            out['report']['Sparsity'] = out['report']['Sparsity'].tolist()
            curr.append(out)
        with open(self.result_path, 'w') as fp:
            fp.write(json.dumps(curr))
        return out

    def params_to_text(self, params):
        t = ""
        for key in params:
            t += key.title() + "\n"
            for subkey in params[key]:
                if subkey in ["Sparsity", "Sparsity by Epoch", "state_dict", "Training Loss by Epoch", "Run Time (S)"]:
                    continue
                if params['data']['D'] > 2 and subkey in ['N', 'n']:
                    continue
                if params['model']['regularization_method'] in [0] and \
                            subkey in ['regularization_lambda', 'regularization_method', 'weight_decay']:
                    continue
                if params['model']['regularization_method'] in [1,2] and subkey in ['weight_decay']:
                    continue
                if params['model']['regularization_method'] == 3 and \
                            subkey in ['regularization_lambda', 'regularization_method']:
                    continue
                k, v = subkey, params[key][subkey]
                t += f"-{k}: {v}\n" 
        return t

    def graph_1d_interpolation(self, data, model, out):
        x = np.linspace(-1, 1, 1000).reshape(-1,1)
        y = model.predict(x).flatten()
        fig, ax = plt.subplots(figsize=(6,8), nrows=2, ncols=1)
        cbar = ax[0].scatter(x, y, c=y, cmap="bwr")
        trainx, trainy = data.training_data()
        ax[0].scatter(trainx, trainy, c='black', marker='*')
        fig.colorbar(cbar, ax=ax[0])
        ax[0].set_title("1D Model Interpolation", fontsize=14)
        ax[1].set_title("Model Sparsity", fontsize=14)
        ax[1].set_ylim(0,1)
        barx = np.arange(len(out['report']['Sparsity']))
        ax[1].bar(barx, out['report']['Sparsity'])
        ax[1].set_xticks(barx)
        ax[1].set_xticklabels(barx)
        ax[0].text(
            1.4, .2, self.params_to_text(out),
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax[0].transAxes,
            fontsize=14
        )
        fname = self.descriptive_filename(model, "-final-sparsity.png")
        fig.savefig(fname, bbox_inches="tight", facecolor="w")
        return fname

    def graph_2d_interpolation(self, data, model, out):
        n = 100
        x = np.linspace(-1, 1, n)
        x, y = np.meshgrid(x, x)
        x, y = x.flatten(), y.flatten()
        z = []
        for i in range(len(x)):
            z.append(model.predict(np.array([[x[i], y[i]]])))
        x = x.reshape(n, n)
        y = y.reshape(n, n)
        z = np.array(z).reshape(n, n)
        fig = plt.figure(figsize=(6,8))
        ax = fig.add_subplot(2,1,1, projection='3d')
        ax2 = fig.add_subplot(2,1,2)
        barx = np.arange(len(out['report']['Sparsity']))
        ax2.bar(barx, out['report']['Sparsity'])
        ax2.set_xticks(barx)
        ax2.set_xticklabels(barx)
        cbar = ax.plot_surface(x, y, z, cmap="bwr", linewidth=0, antialiased=False)
        trainx, trainy = data.training_data()
        ax.scatter3D(trainx[:,0], trainx[:,1], trainy, c="black", marker="*")
        fig.colorbar(cbar, ax=ax, shrink=0.5, aspect=5)
        ax.set_title("2D Model Interpolation", fontsize=14)
        ax2.set_title("Model Sparsity", fontsize=14)
        ax.text2D(
            1.5, .2, self.params_to_text(out),
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=14
        )
        fname = self.descriptive_filename(model, "-final-sparsity.png")
        fig.savefig(fname, bbox_inches="tight", facecolor="w")
        return fname

    def graph_Nd_learned_sparsity(self, data, model, out):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f"{data.D}-D Model Sparsity", fontsize=14)
        ax.set_ylim(0,1)
        barx = np.arange(len(out['report']['Sparsity']))
        ax.bar(barx, out['report']['Sparsity'])
        ax.set_xticks(barx)
        ax.set_xticklabels(barx)
        ax.text(
            1.1, .5, self.params_to_text(out),
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=14
        )
        fname = self.descriptive_filename(model, "-final-sparsity.png")
        fig.savefig(fname, bbox_inches="tight", facecolor="w")
        return fname

    def graph_sparsity_by_epoch(self, data, model, out):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f"{data.D}-D Average Sparsity by Training Epoch", fontsize=14)
        ax.set_ylim(0,1)
        epochs = []
        avgs = []
        for foo in out['report']['Sparsity by Epoch']:
            epochs.append(foo[0])
            avgs.append(np.mean(foo[1]))
        ex = np.arange(len(epochs))
        ax.bar(ex, avgs)
        ax.set_xticks(ex)
        ax.set_xticklabels(epochs, rotation=45)
        ax.text(
            1.1, .5, self.params_to_text(out),
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=14
        )
        fname = self.descriptive_filename(model, "-training-sparsity.png")
        fig.savefig(fname, bbox_inches="tight", facecolor="w")
        return fname

    def save_state_dict(self, model):
        fname = self.descriptive_filename(model, ".pt")
        torchsave(model.state_dict(), fname)
        return fname

    def save_state_dict_matlab(self, model):
        out = {}
        sd = model.state_dict()
        for i in range(model.layers):
            key = 'blocks.{}.{}.weight'
            w = sd[key.format(i, 'W')].numpy()
            v = sd[key.format(i, 'V')].numpy()
            s = sd[key.format(i, 'skip_l')].numpy()
            out[f'W_{i}'] = w
            out[f'V_{i}'] = v
            out[f'S_{i}'] = s
        fname = self.descriptive_filename(model, ".mat")
        savemat(fname, out)
        return fname
    
    def descriptive_filename(self, model, ext):
        fname = f"{self.results_dir}/D={model.D}"
        fname += f"_R={model.relu_width}_L={model.linear_width}"
        fname += f"_LR={model.learning_rate}_WD={model.weight_decay}"
        fname += f"_Term={model.regularization_method}_Layers={model.layers}"
        fname += f"_Lam={model.regularization_lambda}_E={model.epochs}"
        fname += ext
        return fname

class ResultsViewer:

    def __init__(self, results_dir):
        self.results_dir = results_dir
        with open(self.results_dir + "/results.json", "r") as fp:
            self.contents = json.loads(fp.read())

    def __getitem__(self, i):
        item = self.contents[i].copy()
        return item

    def __len__(self):
        return len(self.contents)

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
    linear_widths = [data.D*data.D]  # *data.n]
    layers = [2] 
    lambdas = [0.001,  0.01, 0.1]
    terms = [1, 2, 3, 4]


    pool = []
    mid = 0
    for rw, lw, lay, e, lam, term in product(relu_widths, linear_widths, \
                                            layers, epochs, lambdas, terms):
        params = {"relu_width": rw, "linear_width": lw, 
        "layers": lay, "epochs": e, "learning_rate": 1e-3,
        "regularization_lambda": lam, "regularization_method": term,
        "weight_decay": lam, "modelid": mid}
        print(params)
        exp = Experiment(data, params)
        pool.append(exp.run)
        mid += 1

    savefns = manager.distributed_run(pool)
    for fn in savefns:
        print(fn())
