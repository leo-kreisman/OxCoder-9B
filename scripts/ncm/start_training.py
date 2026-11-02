"""Finite, supervised response-distillation and Muon codec continuation job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--data',default='results/ncm-corpus-v2')
    p.add_argument('--init-checkpoint',default='results/ncm-byte-codec-v3/best.pt')
    p.add_argument('--teacher-file',required=True)
    p.add_argument('--server',required=True)
    p.add_argument('--port',type=int,default=18119)
    p.add_argument('--per-source',type=int,default=32)
    p.add_argument('--steps',type=int,default=12000)
    a=p.parse_args()
    if a.steps<1 or a.per_source<1:p.error('Counts must be positive')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(out).free < 3*1024**3:
        raise RuntimeError('At least 3 GiB free disk required')
    for path in (a.teacher_file,a.server,a.init_checkpoint):
        if not Path(path).is_file():raise FileNotFoundError(path)
    # Never accidentally connect the teacher client to an existing unrelated server.
    with socket.socket() as sock:sock.bind(('127.0.0.1',a.port))
    status={'pid':os.getpid(),'started_unix':time.time(),'configuration':vars(a),
            'phase':'starting','quality_ready':False,'generator_training_started':False,
            'implementation_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted(Path('scripts/ncm').glob('*.py'))}}
    def update(phase,**extra):
        status.update(phase=phase,updated_unix=time.time(),**extra)
        temp=out/'status.tmp';temp.write_text(json.dumps(status,indent=2)+'\n')
        temp.replace(out/'status.json');print(json.dumps({'phase':phase,**extra}),flush=True)
    teacher=None;child=None
    def stop(proc):
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
    def interrupted(signum,frame):raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def run(phase,module,*args):
        nonlocal child
        cmd=[sys.executable,'-u','-m',module,*map(str,args)]
        update(phase,command=cmd)
        with (out/(phase+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            code=child.wait()
        if code:raise RuntimeError(f'{phase} exited {code}; see {phase}.log')
    try:
        with (out/'teacher.log').open('w') as log:
            cmd=[a.server,'-m',a.teacher_file,'--host','127.0.0.1','--port',str(a.port),
                 '-c','4096','-np','1','-ngl','99','-fa','on','--reasoning','off',
                 '--alias','oxcoder-ncm-teacher']
            teacher=subprocess.Popen(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'},
                                     stdout=log,stderr=subprocess.STDOUT)
            update('loading_teacher',teacher_pid=teacher.pid,teacher_command=cmd)
            for _ in range(240):
                if teacher.poll() is not None:raise RuntimeError('Teacher exited during loading')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health',timeout=2) as r:
                        if r.status==200:break
                except (OSError,TimeoutError):pass
                time.sleep(1)
            else:raise TimeoutError('Teacher startup timed out')
            run('distillation','scripts.ncm.distill','--data',a.data,'--output',out/'teacher',
                '--teacher-file',a.teacher_file,'--url',f'http://127.0.0.1:{a.port}',
                '--per-source',a.per_source,'--max-tokens',1536,'--prompt-limit',2400)
        stop(teacher)
        manifest=json.loads((out/'teacher/manifest.json').read_text())
        accepted=[r for r in manifest['records'] if r['accepted_complete']]
        sources={r['source'] for r in accepted}
        expected=set(json.loads((Path(a.data)/'manifest.json').read_text())['sources'])
        if sources!=expected:raise RuntimeError('No complete teacher response for each source')
        update('preparing_data',accepted_teacher_responses=len(accepted),teacher_sources=sorted(sources))
        run('data','scripts.ncm.byte_followup','--data',a.data,'--teacher',out/'teacher',
            '--output',out/'data')
        run('codec_training','scripts.ncm.train','codec','--data',out/'data',
            '--output',out/'codec','--init-checkpoint',a.init_checkpoint,
            '--optimizer','muon','--lr',.0001,'--batch',16,'--steps',a.steps,
            '--eval-every',500,'--devices','cuda:1')
        run('complete_response_evaluation','scripts.ncm.evaluate_documents',
            '--checkpoint',out/'codec/best.pt','--data',a.data,'--output',out/'complete-gate.json')
        gate=json.loads((out/'complete-gate.json').read_text())
        update('codec_run_finished',codec_gate_passed=gate['passed'],
               next_stage='generator admission review' if gate['passed'] else 'codec fidelity still blocks generator',
               quality_ready=False)
    except BaseException as e:
        update('failed',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        stop(child);stop(teacher)


if __name__=='__main__':main()
