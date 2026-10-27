"""Linux-only isolated benchmark launcher; swap disabled, no global cache flushes.
Usage: python run_limited.py --gpu 1 --report run.resources.json -- python probe.py ...
The selected physical GPU must be free of other inference jobs. Its entire used
memory (including unrelated graphics allocations) is conservatively counted.
"""
import argparse,json,os,subprocess as sp,time,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int);p.add_argument('--report',type=Path,required=True)
p.add_argument('--seconds',type=int,default=180);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
command=a.command[1:] if a.command[:1]==['--'] else a.command
if not command: p.error('command required after --')
unit='kindi-q4-'+uuid.uuid4().hex[:10]
limit=1_700_000_000 if a.gpu is not None else 8_000_000_000
properties=['MemoryMax='+str(limit),'MemorySwapMax=0','Nice=15','IOSchedulingClass=idle','CPUQuota=200%','TasksMax=64','RuntimeMaxSec='+str(a.seconds)]
args=['systemd-run','--user','--unit='+unit,'--working-directory='+str(Path.cwd())]
for prop in properties:args+=['-p',prop]
# Forward explicit runtime settings; do not serialize credentials or unrelated environment.
env={k:v for k,v in os.environ.items() if k.startswith(('KINDI_','OMP_','CUPY_')) or k in ('PATH','PYTHONPATH','CUDA_PATH','OPENBLAS_NUM_THREADS')}
if a.gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(a.gpu)
sp.run(args+['/usr/bin/env']+[k+'='+v for k,v in env.items()]+command,check=True)
samples=[];failure=None
try:
 deadline=time.monotonic()+a.seconds+10
 while time.monotonic()<deadline:
  state=sp.check_output(['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True).strip()
  if state not in ('active','activating'):break
  cg=sp.check_output(['systemctl','--user','show',unit,'-p','ControlGroup','--value'],text=True).strip()
  path=Path('/sys/fs/cgroup')/cg.lstrip('/')
  try:host=int((path/'memory.current').read_text())
  except FileNotFoundError:time.sleep(.1);continue
  gpu=0
  if a.gpu is not None:
   gpu=int(sp.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())*1024**2
  info=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())
  available=int(info['MemAvailable'].split()[0])*1024
  total=int(info['MemTotal'].split()[0])*1024
  samples.append(dict(time=time.time(),host_cgroup=host,gpu_physical=gpu,total=host+gpu))
  if host+gpu>8_000_000_000 or available<min(12_000_000_000,total//4):failure='memory guard';break
  time.sleep(.5)
 else:failure='timeout'
except BaseException:
 sp.run(['systemctl','--user','stop',unit]);raise
finally:
 if failure:sp.run(['systemctl','--user','stop',unit])
 result=sp.check_output(['systemctl','--user','show',unit,'-p','Result','-p','ExecMainStatus'],text=True)
 log=sp.check_output(['journalctl','--user','-u',unit,'--no-pager','-o','cat'],text=True)
 a.report.parent.mkdir(parents=True,exist_ok=True)
 a.report.with_suffix('.log').write_text(log)
 a.report.write_text(json.dumps(dict(unit=unit,result=result,failure=failure,peak_total=max((x['total'] for x in samples),default=0),samples=samples),indent=2)+'\n')
 print(log)
 if failure or 'ExecMainStatus=0' not in result:raise SystemExit(1)
