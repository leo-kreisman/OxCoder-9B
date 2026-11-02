"""Run the finite open-data codec ablation and downstream NCM diagnostics."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from .open_corpus import file_sha


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=a.resume)
    lock = (out/'.pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    a.owns_lock = True
    sources = [Path(__file__),Path(__file__).with_name('open_baseline.py'),Path(__file__).with_name('open_corpus.py')]
    hashes = {p.name:file_sha(p) for p in sources}
    snapshot = out/'implementation'
    snapshot.mkdir(exist_ok=True)
    if (snapshot/'manifest.json').exists():
        if json.loads((snapshot/'manifest.json').read_text()) != hashes:
            raise ValueError('Pipeline source changed; use a new experiment directory')
    else:
        for p in sources:
            (snapshot/p.name).write_bytes(p.read_bytes())
        write(snapshot/'manifest.json',hashes)
    state = {'phase':'waiting_for_corpus','quality_ready':False,'complete':False}
    write(out/'pipeline.json',state)
    deadline = time.monotonic()+4*3600
    corpus = Path(a.corpus)
    while not (corpus/'manifest.json').exists():
        status = subprocess.run(['systemctl','--user','is-active',a.acquisition_unit], capture_output=True, text=True)
        if status.stdout.strip() not in ('active','activating') or time.monotonic() > deadline:
            raise RuntimeError('Corpus acquisition is not active or exceeded wait budget')
        time.sleep(15)
    manifest = json.loads((corpus/'manifest.json').read_text())
    if not manifest['token_budget_reached'] or min(manifest['stats'][s+'_documents'] for s in ('train','validation','test'))<1000:
        raise RuntimeError('Corpus budget/evaluation coverage gate failed')
    data = out/'data'
    def phase(name, args):
        if any(file_sha(p)!=hashes[p.name] for p in sources):
            raise RuntimeError('Source changed during pipeline execution')
        state['phase'] = name
        write(out/'pipeline.json',state)
        command = [sys.executable,'-u','-m','scripts.ncm.open_baseline',*args]
        write(out/(name+'-command.json'),command)
        with (out/(name+'.log')).open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not (data/'manifest.json').exists():
        phase('prepare',['prepare','--corpus',str(corpus),'--output',str(data)])
    summaries = {}
    for name, mix in [('code',0.),('mixed',.2)]:
        c = out/('codec-'+name)
        if not (c/'progress.json').exists() or not json.loads((c/'progress.json').read_text())['training_complete']:
            args = ['codec','--data',str(data),'--output',str(c),'--assistant-fraction',str(mix),
                    '--steps',str(a.codec_steps),'--batch','8','--accumulation','8','--device',a.device]
            if (c/'latest.pt').exists():
                args.append('--resume')
            phase('codec-'+name,args)
        metric = json.loads((c/'selected-evaluation.json').read_text())
        summaries[name] = {'codec': {k:v for k,v in metric.items() if k!='rows'}}
        # This is a diagnostic even for imperfect codecs, never promotion to a
        # larger model. Complete collapse makes a predictor experiment pointless.
        if metric['unique_codes_per_level'][-1] < 2:
            summaries[name]['generator_skipped'] = 'finest codebook collapsed'
            continue
        g = out/('generator-'+name)
        if not (g/'progress.json').exists() or not json.loads((g/'progress.json').read_text())['training_complete']:
            args = ['generator','--data',str(data),'--codec',str(c/'best.pt'),'--output',str(g),
                    '--assistant-fraction','.2','--steps',str(a.generator_steps),
                    '--batch','4','--accumulation','8','--device',a.device]
            if (g/'latest.pt').exists():
                args.append('--resume')
            phase('generator-'+name,args)
        summaries[name]['generator'] = {k:v for k,v in json.loads((g/'selected-evaluation.json').read_text()).items() if k!='rows'}
        coding = json.loads((g/'coding-development.json').read_text())
        summaries[name]['coding'] = {k:v for k,v in coding.items() if k!='tasks'}
    # Report comparable output measures; concept accuracy is codec-dependent.
    eligible = {k:v for k,v in summaries.items() if 'generator' in v}
    winner = max(eligible,key=lambda k:(eligible[k]['coding']['passed'],
        eligible[k]['generator']['exact_windows'],eligible[k]['generator']['mean_free_nonpad_token_accuracy'])) if eligible else None
    write(out/'comparison.json', {'arms':summaries,'development_winner':winner,
        'winner_is_qualified_model':False,'quality_ready':False,
        'corpus_manifest_sha256':file_sha(corpus/'manifest.json'),
        'decision':'Compare output fidelity and executable development tests before planning donor transfer; no automatic 9B training or release.'})
    state.update(phase='complete',complete=True)
    write(out/'pipeline.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--acquisition-unit',default='ncm-open-corpus-v11c')
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--codec-steps',type=int,default=6000)
    p.add_argument('--generator-steps',type=int,default=3000)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    try:
        run(a)
    except Exception as e:
        if getattr(a, 'owns_lock', False):
            write(Path(a.output)/'failure.json',{'error':str(e),'quality_ready':False})
        raise
