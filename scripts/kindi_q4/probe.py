"""Bounded end-to-end Q4 Kindi probe; records real generation times and tokens."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from kindi.engine import make_engine
from kindi.tokenizer import Tok

p=argparse.ArgumentParser();p.add_argument('--pack',required=True);p.add_argument('--prompt-file',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,default=16)
p.add_argument('--context',type=int,default=512);a=p.parse_args()
if not 1<=a.tokens<=128:raise ValueError('Bounded token count required')
begin=time.monotonic();e=make_engine(a.pack,ctx=a.context);loaded=time.monotonic()
tok=Tok(e.vocab,str(Path(a.pack)/'merges.txt'));ids=tok.encode(a.prompt_file.read_text())
if not ids or len(ids)+a.tokens>=a.context:raise ValueError('Prompt/context mismatch')
out=[];times=[]
for i,t in enumerate(ids[:-1]):e.feed(t,i)
next_token=e.step(ids[-1],len(ids)-1);times.append(time.monotonic());out.append(int(next_token))
logits=e.ylog[:e.ylog_n]
np.save(str(a.output)+'.first-logits.npy',logits.get() if hasattr(logits,'get') else logits)
for pos in range(len(ids),len(ids)+a.tokens-1):
 if next_token in (248044,248046): break
 next_token=e.step(next_token,pos);times.append(time.monotonic());out.append(int(next_token))
 print('token',len(out),'elapsed',round(times[-1]-loaded,2),flush=True)
text=b''.join(tok.decode_bytes(t) for t in out).decode('utf-8',errors='replace')
report=dict(prompt_ids=ids,token_ids=out,text=text,load_seconds=loaded-begin,
 ttft_seconds=times[0]-loaded,decode_tps=(len(out)-1)/(times[-1]-times[0]) if len(out)>1 else None,
 completion_seconds=times[-1]-loaded,stream=os.environ.get('KINDI_Q4_STREAM','0'),
 context=a.context,threads=os.environ.get('OMP_NUM_THREADS','2'))
import resource,sys
report['max_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
if hasattr(e,'peak_combined'):
 e._memory_check()
 report.update(gpu_host_peak_sampled=e.peak_combined,weight_cache_hits=e.w.hits,
  weight_cache_misses=e.w.misses,logical_weight_reads=e.w.read_bytes,gpu_cache_limit=e.w.limit)
try:
 name=next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
 cg=Path('/sys/fs/cgroup')/name.lstrip('/')
 report['cgroup']={k:(cg/k).read_text().strip() for k in ['memory.max','memory.peak','memory.swap.current','memory.events']}
 report['process_io']=Path('/proc/self/io').read_text()
except FileNotFoundError:pass
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
