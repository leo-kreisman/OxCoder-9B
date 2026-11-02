"""Run isolated coding evaluation after a finite generator job finishes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit',required=True);p.add_argument('--run',required=True)
    p.add_argument('--source',required=True);p.add_argument('--codec',required=True)
    p.add_argument('--wait-seconds',type=int,default=10800);a=p.parse_args();run=Path(a.run)
    run.mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):
        value={'phase':phase,'time_unix':time.time(),'quality_ready':False,**extra}
        tmp=run/'evaluation-status.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n')
        tmp.replace(run/'evaluation-status.json');print(json.dumps(value),flush=True)
    try:
        status('waiting_for_generator')
        deadline=time.monotonic()+a.wait_seconds
        while True:
            state=subprocess.run(['systemctl','--user','show',a.unit,'-p','ActiveState','--value'],
                                 capture_output=True,text=True,check=True).stdout.strip()
            if state not in ('active','activating','deactivating'):break
            if time.monotonic()>deadline:raise TimeoutError('Generator wait limit exceeded')
            time.sleep(10)
        training=json.loads((run/'training.json').read_text())
        checks=json.loads((run/'structural-validation.json').read_text())
        if not training['complete'] or not checks['checkpoint_reload_exact']:
            raise RuntimeError('Generator did not complete and verify its checkpoint')
        status('coding_evaluation')
        cmd=[sys.executable,'-u','-m','scripts.ncm.coding_eval','--source',a.source,'--codec',a.codec,
             '--checkpoint',str(run/'latest.pt'),'--output',str(run/'coding-evaluation.json')]
        with (run/'coding-evaluation.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((run/'coding-evaluation.json').read_text())
        status('coding_evaluation_finished',passed_tasks=report['passed'],evaluated_tasks=report['evaluated'],
               limitation='Elementary Python diagnostics only; professional quality and 8GB deployment remain unverified')
    except BaseException as e:
        status('failed',error=f'{type(e).__name__}: {e}');raise


if __name__=='__main__':main()
