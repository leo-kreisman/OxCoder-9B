"""Finish complete-response features, then compare two matched native NCM runs.

Both arms use the same corpus, initialization, random seed, dropout and adaptive
residual policy. Only decoded-text auxiliary loss differs. Completion is never
acceptance; unchanged executable tests and a fresh-process replay remain required.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from .backbone import digest
from .run_context_stage import active,read
from .context_cache import merge
from .complete_context_cache import run as complete_cache
from .monitor_context_comparison import run as monitor


def run(a):
    root=Path(a.root).resolve();out=root/'results/ncm-complete-pipeline-v28';out.mkdir(exist_ok=True)
    def status(phase,**values):
        data={'phase':phase,'time':time.time(),'quality_ready':False,**values}
        tmp=out/'progress.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(out/'progress.json')
        print(json.dumps(data),flush=True)
    def launch(name,arguments,memory='12G'):
        if active(name+'.service'):return
        subprocess.run(['systemctl','--user','reset-failed',name+'.service'],capture_output=True)
        subprocess.run(['systemd-run','--user','--unit='+name,'--collect','--property=WorkingDirectory='+str(root),
            '--property=MemoryMax='+memory,'--property=MemorySwapMax=0','--property=TasksMax=128','--property=RuntimeMaxSec=43200',
            '--property=StandardOutput=append:'+str(root/'results'/f'{name}.log'),
            '--property=StandardError=append:'+str(root/'results'/f'{name}.log'),
            '--setenv=PYTHONPATH=results/ncm-kernels:results/ncm-deps:results/ncm-data-deps','--setenv=OMP_NUM_THREADS=2',
            '--setenv=TOKENIZERS_PARALLELISM=false','--setenv=HF_HUB_OFFLINE=1',
            '--setenv=PYTORCH_ALLOC_CONF=expandable_segments:True',a.python,*arguments],check=True)
    try:
        audit=read(a.audit)
        if (not audit.get('complete') or not audit.get('passed') or
                any(digest(root/'scripts/ncm'/name)!=checksum for name,checksum in audit['implementation_sha256'].items())):
            raise ValueError('Current decoded objective lacks full-shape numerical/memory qualification')
        identity={'init_checkpoint_sha256':digest(a.init),'selection_sha256':digest(root/'results/ncm-complete-selection-v27/selection.pt'),
            'objective_audit_sha256':digest(a.audit),'planned_steps':a.steps,'seed':28001,
            'comparison':'decoded byte CE weight 0 versus 0.1; both adaptive policy .5 and dropout .1',
            'implementation_sha256':digest(__file__),'quality_ready':False}
        if (out/'identity.json').exists():
            if read(out/'identity.json')!=identity:raise ValueError('Pipeline identity changed')
        else:(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        units=[f'ncm-complete-features-v27-{i}' for i in (0,1)]
        status('waiting_for_complete_response_features')
        while any(active(unit+'.service') for unit in units):time.sleep(20)
        if not all(read(root/'results'/unit/'manifest.json').get('complete') for unit in units):
            raise RuntimeError('Feature extraction failed or is incomplete')
        merged=out/'all-features.json';qualified=out/'features.json'
        if not merged.exists():
            status('checking_and_merging_features')
            merge(root/'results/ncm-complete-selection-v27',[root/'results'/unit for unit in units],merged)
        if not qualified.exists():
            complete_cache(SimpleNamespace(input=str(merged),selection=str(root/'results/ncm-complete-selection-v27'),output=str(qualified)))
        if not read(qualified).get('complete') or not read(qualified)['identity'].get('complete_training_responses_only'):
            raise ValueError('Training cache not qualified')
        names=['ncm-context-control-v28','ncm-context-decoded-v28']
        status('training_matched_complete_response_arms',feature_manifest_sha256=digest(qualified))
        for device,(name,weight) in enumerate(zip(names,['0','0.1'])):
            folder=root/'results'/name
            if (folder/'completed.json').exists() or (folder/'stopped.json').exists():continue
            arguments=['-m','scripts.ncm.train_context_corpus','--cache',str(qualified),'--codec',a.codec,
                '--init',a.init,'--output',str(folder),'--device',f'cuda:{device}','--byte-loss',weight,'--steps',str(a.steps)]
            if (folder/'latest.pt').exists():arguments.append('--resume')
            launch(name,arguments)
        monitor(SimpleNamespace(root=str(root),output=str(out/'comparison'),python=a.python,source=a.source,codec=a.codec,
            folders=[str(root/'results'/n) for n in names],training_units=[n+'.service' for n in names],
            device='cuda:1',eval_unit='ncm-context-evaluation-v28'))
        status('comparison_stopped',acceptance=read(out/'comparison/acceptance.json'))
    except BaseException as error:
        status('error',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','python','source','codec','init','audit'):p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=10000);run(p.parse_args())
