"""Continue an initialized research generator through cache audit and training.

No release promotion is performed. The exact-target curriculum can be used
despite a failed broad codec gate, with all evaluation failures retained.
"""
import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from .backbone import digest
from .train import write_json


def validate_initialization(root,codec_sha):
    root=Path(root)
    training=json.loads((root/'training.json').read_text())
    structure=json.loads((root/'structural-validation.json').read_text())
    audit=json.loads((root/'backbone-audit.json').read_text())
    if not training['complete']:raise ValueError('Initialization did not complete')
    if not structure['checkpoint_reload_exact'] or not structure['all_32_adapters_receive_gradients']:
        raise ValueError('Initialization structural checks failed')
    if not audit['passed']:raise ValueError('Source-backbone audit failed')
    if training['identity']['codec_sha256']!=codec_sha:raise ValueError('Initialization uses a different codec')
    return training['identity']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['unit','initial-run','codec','source','data','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--steps',type=int,default=12000)
    p.add_argument('--wait-seconds',type=int,default=3600)
    a=p.parse_args()
    if a.steps<1 or a.wait_seconds<1:raise ValueError('Positive step/wait limits required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    child=None
    def status(phase,**extra):
        record={'phase':phase,'time_unix':time.time(),'quality_ready':False,**extra}
        write_json(out/'status.json',record);print(json.dumps(record),flush=True)
    def interrupted(signum,frame):raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def run(phase,module,*args):
        nonlocal child
        command=[sys.executable,'-u','-m',module,*map(str,args)]
        status(phase,command=command)
        with (out/(phase+'.log')).open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            code=child.wait()
        child=None
        if code:raise RuntimeError(f'{phase} failed with exit {code}; inspect its log')
    try:
        status('waiting_for_initialization',unit=a.unit)
        deadline=time.monotonic()+a.wait_seconds
        while True:
            state=subprocess.run(['systemctl','--user','show',a.unit,'-p','ActiveState','--value'],
                                 capture_output=True,text=True,check=True).stdout.strip()
            if state not in ('active','activating','deactivating'):break
            if time.monotonic()>deadline:raise TimeoutError('Initialization wait limit exceeded')
            time.sleep(10)
        codec_sha=digest(a.codec);identity=validate_initialization(a.initial_run,codec_sha)
        codec=out/'codec.pt';initial=out/'initial.pt'
        shutil.copyfile(a.codec,codec)
        source_initial=Path(a.initial_run)/'latest.pt';initial_sha=digest(source_initial)
        shutil.copyfile(source_initial,initial)
        if digest(codec)!=codec_sha or digest(initial)!=initial_sha:raise RuntimeError('Artifact changed during snapshot')
        write_json(out/'lineage.json',{'source_sha256':identity['source_sha256'],'codec_sha256':codec_sha,
            'initial_sha256':initial_sha,'scope':'frozen contextual codec; new exact-target generator experiment',
            'quality_ready':False})
        audit=out/'prefix-audit.json'
        run('prefix_audit','scripts.ncm.audit_prefix','--source',a.source,'--codec',codec,
            '--checkpoint',initial,'--output',audit,'--full-precision-reductions')
        check=json.loads(audit.read_text())
        if not check.get('passed') or not check.get('complete'):raise RuntimeError('Prefix audit did not pass')
        run('generator_training','scripts.ncm.train_geometry','--source',a.source,'--codec',codec,
            '--init',initial,'--data',a.data,'--prefix-audit',audit,'--output',out/'generator',
            '--steps',a.steps,'--head-warmup',100,'--eval-every',250,'--save-every',25,
            '--head-input-norm','--calibrate-head','--nearest-code','--residual-byte-loss',1.0)
        report=json.loads((out/'generator'/f'coding-{a.steps}.json').read_text())
        status('training_finished',passed_tasks=report['passed'],evaluated_tasks=report['evaluated'],
               limitation='Development evaluation only; broad codec, fresh coding and deployment qualification still required')
    except BaseException as error:
        status('failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
