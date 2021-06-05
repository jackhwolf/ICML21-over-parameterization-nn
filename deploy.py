import os
import libtmux
import asyncio
from dask.distributed import Scheduler, Worker, Client
from contextlib import AsyncExitStack
import yaml
from sklearn.model_selection import ParameterGrid
import time

class TmuxDeployer:

    def __init__(self, session_name='ICML'):
        self.sname = session_name
        self.session = libtmux.Server().new_session(self.sname, kill_session=True)

    def __call__(self, experiment_fname, workers, input_fname, addr):
        experiment_fname = os.path.abspath(experiment_fname)
        pane = self.session.attached_window.attached_pane
        pane.send_keys('source venv/bin/activate')
        keys = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', \
                'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS']
        for key in keys:
            pane.send_keys(f'export {key}=1')
        cmd = f"python3 {experiment_fname} {workers} {input_fname} {addr}"
        pane.send_keys(cmd)
        
class DaskManager:

    def __init__(self, workers):
        self.workers = workers

    def distributed_run(self, fnpool, addr):
        client = Client(addr, asynchronous=False)
        files = ["data.py", "model.py", "deploy.py", "experiment.py", "sparsity_experiment.py"]
        for f in files:
            client.upload_file(f)
        futures = []
        for i in range(len(fnpool)):
            futures.append(client.submit(fnpool[i]))
        return futures

class YamlInput:

    def __init__(self, fname):
        self.fname = fname
        with open(self.fname, "r") as fp:
            self.c = yaml.load(fp, Loader=yaml.FullLoader)
        for k, v in self.c['parameters'].items():
            if isinstance(v, list):
                continue
            self.c['parameters'][k] = [v]        
        if not isinstance(self.c['data_id'], list):
            self.c['data_id'] = [self.c['data_id']]

    def iterate_inputs(self):
        grid = list(ParameterGrid(self.c['parameters']))
        data_ids = self.c['data_id']
        for data_id in data_ids:
            no_reg_done = False
            for g in grid:
                if g['regularization_method'] == 'none' and no_reg_done:
                    continue
                yield data_id, g
                if g['regularization_method'] == 'none':
                    no_reg_done = True

    def __getitem__(self, k):
        return self.c[k]    


if __name__ == '__main__':
    import sys
    d = TmuxDeployer()
    d(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

        
