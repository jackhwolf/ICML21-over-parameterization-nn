import os
import libtmux

class Deployer:

    def __init__(self, session_name='ICML'):
        self.sname = session_name
        self.session = libtmux.Server().new_session(self.sname, kill_session=True)

    def __call__(self, fname, workers):
        fname = os.path.abspath(fname)
        cmd = f"python3 {fname} {workers}"
        pane =  self.session.attached_window.attached_pane
        pane.send_keys('source venv/bin/activate')
        keys = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', \
                'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS']
        for key in keys:
            pane.send_keys(f'export {key}=1')
        pane.send_keys(cmd)

if __name__ == '__main__':
    import sys
    d = Deployer()
    d(sys.argv[1], sys.argv[2])

        
