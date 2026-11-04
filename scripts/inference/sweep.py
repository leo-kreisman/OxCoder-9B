"""Sequential OxCoder server sweep; conservative sampled RSS + CUDA allocation cap."""
import argparse,json,os,pathlib,subprocess,time,threading,urllib.request

def request(port,path,payload=None):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(f'http://127.0.0.1:{port}/{path}',data=data,headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(req,timeout=300))

def main():
 p=argparse.ArgumentParser();p.add_argument('--server',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
 configs=[('baseline',['-fa','off']),('flash',['-fa','on']),('flash-small',['-fa','on','-ub','64']),('lookup',['-fa','on','--spec-type','ngram-simple','--spec-ngram-simple-size-n','3','--spec-ngram-simple-size-m','16'])]
 for name,extra in configs:
  report=out/(name+'.json')
  if report.exists():continue
  cmd=[a.server,'-m',a.model,'--host','127.0.0.1','--port','18197','--load-mode','none','--no-host','-ngl','99','--split-mode','none','-c','4096','-np','1','-b','256','-ub','128','-t','4',*extra]
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'};record={'configuration':name,'command':cmd,'budget_bytes':8*1024**3,'scope':'Linux CUDA; sampled process RSS plus process CUDA allocation; not Mac measurement','samples':[],'peak_bytes':0,'budget_exceeded':False}
  with (out/(name+'.log')).open('w') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=log);done=threading.Event()
   def monitor():
    while not done.is_set() and proc.poll() is None:
     try:
      status=pathlib.Path(f'/proc/{proc.pid}/status').read_text();rss=int(next(x.split()[1] for x in status.splitlines() if x.startswith('VmRSS:')))*1024
      gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
      used=sum(int(line.split(',')[1])*1024**2 for line in gpu.splitlines() if line.split(',')[0].strip()==str(proc.pid))
      record['peak_bytes']=max(record['peak_bytes'],rss+used)
      if rss+used>record['budget_bytes']:record['budget_exceeded']=True;proc.terminate();return
     except (OSError,ValueError,StopIteration,subprocess.SubprocessError):pass
     done.wait(.25)
   thread=threading.Thread(target=monitor);thread.start()
   try:
    for _ in range(300):
     if proc.poll() is not None:raise RuntimeError('server exited before readiness')
     try:request(18197,'health');break
     except Exception:time.sleep(1)
    else:raise TimeoutError('server readiness')
    prompt='Return only Python code. Implement a stable merge sort for a list of integers, with type annotations and a brief docstring.\n'
    for repeat in range(3):
     start=time.monotonic();r=request(18197,'completion',{'prompt':prompt,'n_predict':256,'temperature':0,'seed':123,'cache_prompt':False,'ignore_eos':False})
     record['samples'].append({'repeat':repeat,'wall_seconds':time.monotonic()-start,'timings':r.get('timings'),'content':r.get('content'),'tokens_predicted':r.get('tokens_predicted')})
   except Exception as e:record['error']=str(e)
   finally:
    done.set()
    if proc.poll() is None:proc.terminate()
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    thread.join();record['exit_code']=proc.returncode;report.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('samples','command')}),flush=True)
if __name__=='__main__':main()
