"""Finite, resumable codec/generator training with explicit quality gates."""
import argparse
import dataclasses
import json
import random
import time
from pathlib import Path
import psutil
import torch
from torch.nn import functional as F
from .codec import Codec, CodecConfig, encode, decode, PAD, EOS
from .backbone import digest


def optimizer_for(module,a,foreach=None):
    if getattr(a,'optimizer','adamw')=='adamw':
        return torch.optim.AdamW((p for p in module.parameters() if p.requires_grad),lr=a.lr,foreach=foreach)
    from .anvil import make_optimizers,OptimizerBundle
    return OptimizerBundle(make_optimizers(module,'muon',a.lr,0.,muon_lr=a.muon_lr,weight_decay=.01))


def record_optimizer(identity,a):
    if getattr(a,'optimizer','adamw')=='muon':
        identity['optimizer']={'name':'torch-Muon+AdamW','torch':str(torch.__version__),
            'muon_lr':a.lr if a.muon_lr is None else a.muon_lr,'adjust_lr_fn':'match_rms_adamw',
            'momentum':.95,'ns_steps':5,'weight_decay':.01}


def initialize_codec(codec, path):
    """Load weights only; changing corpus/optimizer starts a new experiment."""
    ck = torch.load(path, weights_only=True, map_location='cpu')
    config = ck.get('config', ck.get('identity', {}).get('config'))
    if config is None or CodecConfig(**config) != codec.config:
        raise ValueError('Initialization codec configuration differs')
    codec.load_state_dict(ck['model'], strict=True)
    return {'checkpoint_sha256':digest(path), 'source_step':ck['step'],
            'scope':'weights only; optimizer and update count reset'}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def cpu_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x, list):
        return [cpu_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(cpu_tree(v) for v in x)
    return x


def admission(devices, cap_gib=10):
    if psutil.virtual_memory().available < 12 << 30:
        raise RuntimeError('Keep at least 12 GiB host RAM available')
    result = {'host_available':psutil.virtual_memory().available, 'devices':[]}
    for d in devices:
        if d.startswith('cuda'):
            free, total = torch.cuda.mem_get_info(d)
            cap = min(cap_gib*(1<<30), free-(4<<30))
            if cap <= 0:
                raise RuntimeError(f'No training room on {d} after 4 GiB reserve')
            torch.cuda.set_per_process_memory_fraction(cap/total, d)
            result['devices'].append({'device':d, 'free':free, 'allocator_cap':cap})
    return result


def headroom(devices):
    if psutil.virtual_memory().available < 12 << 30:
        raise RuntimeError('Host memory reserve exhausted; checkpoint retained')
    for d in devices:
        if d.startswith('cuda') and torch.cuda.mem_get_info(d)[0] < 3 << 30:
            raise RuntimeError(f'GPU reserve exhausted on {d}; checkpoint retained')


def load_data(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    if digest(path/'examples.jsonl') != manifest['examples_sha256']:
        raise ValueError('Dataset checksum mismatch')
    rows = [json.loads(line) for line in (path/'examples.jsonl').read_text().splitlines()]
    groups = {}
    for row in rows:
        if row['group'] in groups and groups[row['group']] != row['split']:
            raise ValueError('Conversation leaks between splits')
        groups[row['group']] = row['split']
    return rows, manifest


def require_complete_gate(path,checkpoint_sha,manifest):
    if not path:
        raise RuntimeError('Complete-response reconstruction report required before generator quality training')
    gate=json.loads(Path(path).read_text())
    corpus=manifest.get('complete_corpus_sha256')
    if not corpus or gate.get('corpus_sha256')!=corpus or gate.get('checkpoint_sha256')!=checkpoint_sha:
        raise ValueError('Complete-response gate does not match the selected codec/corpus')
    if not gate.get('passed') or not all(gate.get(k,{}).get('passed') for k in ('validation','probes')):
        raise RuntimeError('Complete-response codec reconstruction failed; generator quality training blocked')


@torch.no_grad()
def exact_training_curriculum(codec,prepared,device):
    """Filter training examples only. Never filter evaluation errors away."""
    training=[(r,c) for r,c in prepared if r['split']=='train'];kept=[];rejected=[]
    codec.eval()
    for start in range(0,len(training),16):
        batch=training[start:start+16]
        ids=torch.tensor([encode(r['target'],codec.config.length) for r,c in batch],device=device)
        codes,_=codec.targets(ids);got=codec.reconstruct_codes(codes).argmax(-1)
        exact=((got==ids)|(ids==PAD)).all(-1).tolist()
        for item,passed in zip(batch,exact):
            if passed:kept.append(item)
            else:rejected.append(item[0]['id'])
    if not kept:raise RuntimeError('No exactly reconstructed training targets')
    report={'training_candidates':len(training),'training_kept':len(kept),'rejected_ids':rejected,
            'evaluation_filtered':False,'quality_ready':False,
            'scope':'training curriculum only; broad reconstruction and generated-code quality remain separate'}
    report['sources']={s:sum(r['source']==s for r,c in kept) for s in {r['source'] for r,c in training}}
    report['distilled_training_kept']=sum(r.get('distilled',False) for r,c in kept)
    if any(n==0 for n in report['sources'].values()):raise RuntimeError('Curriculum lost an entire source')
    return kept+[(r,c) for r,c in prepared if r['split']!='train'],report


@torch.no_grad()
def evaluate_candidate(model,codec,prepared,device):
    selected=[]
    for source in sorted({r['source'] for r,c in prepared}):
        candidates=[(r,c) for r,c in prepared if r['source']==source and r['split']=='validation']
        selected.extend(candidates[:2])
        ending=next(((r,c) for r,c in candidates if c['is_final_chunk'] and r['id'] not in {x['id'] for x,_ in selected}),None)
        if ending is not None:selected.append(ending)
    was_training=model.training;model.eval();records=[]
    try:
        for row,c in selected:
            ids=torch.tensor([encode(row['target'],codec.config.length)],device=device)
            codes,inputs=codec.targets(ids);prompt=torch.tensor([c['ids']],dtype=torch.long)
            prefix_state=model.prepare_prefix(prompt)
            losses=[];accuracies=[]
            for level in range(len(codes)):
                logits=model(inputs[level],prompt,level,prefix_state=prefix_state);target=codes[level].to(logits.device)
                losses.append(float(F.cross_entropy(logits.flatten(0,1),target.flatten())))
                accuracies.append(float((logits.argmax(-1)==target).float().mean()))
            generated,_,stop=model.generate_chunk(codec,prompt)
            try:text=decode(generated[0].tolist());valid=True
            except (ValueError,UnicodeError):text=None;valid=False
            records.append({'id':row['id'],'source':row['source'],'teacher_forced_code_loss':sum(losses)/len(losses),
                'teacher_forced_code_accuracy_by_level':accuracies,'generated_text':text,
                'valid_utf8_and_eos':valid,'exact_generated_chunk':text==row['target'],
                'stop_probability':float(stop[0]),'expected_final_chunk':c['is_final_chunk']})
    finally:model.train(was_training)
    return {'examples':records,'quality_ready':False,
            'scope':'up to three development chunks per source, including an ending; real free-running predictions, not a coding benchmark'}


def decoded_byte_loss(codec, logits, earlier_latents, target):
    """Auxiliary differentiable loss; never supplies target bytes as model inputs."""
    from .codec import resize
    if logits.shape[1] != codec.config.levels[-1]:
        raise ValueError('Byte loss requires the final concept level')
    # The frozen codebook/decoder transmit gradients to generator probabilities.
    expected = logits.softmax(-1) @ codec.books[-1].weight.to(logits)
    cumulative = resize(earlier_latents.to(logits), codec.config.length)
    cumulative = cumulative + resize(expected, codec.config.length)
    decoded = codec.decode_latents(cumulative)
    return F.cross_entropy(decoded.flatten(0,1), target.to(decoded.device).flatten())


def curriculum_buckets(train):
    """Balance represented source, teacher provenance and response-end strata."""
    buckets={}
    for row,condition in train:
        key=(row['source'],bool(row.get('distilled',False)),condition['is_final_chunk'])
        buckets.setdefault(key,[]).append((row,condition))
    return list(buckets.values())


@torch.no_grad()
def reconstruction(codec, rows, device, batch=16):
    codec.eval()
    correct = total = exact = eos_correct = valid_text = 0
    used = [set() for _ in codec.books]
    for start in range(0, len(rows), batch):
        ids = torch.tensor([encode(r['target'], codec.config.length) for r in rows[start:start+batch]], device=device)
        codes, _ = codec.targets(ids)
        predicted = codec.reconstruct_codes(codes).argmax(-1)
        mask = ids != PAD
        correct += int(((predicted == ids)&mask).sum())
        total += int(mask.sum())
        exact += int((((predicted == ids)|~mask).all(-1)).sum())
        eos_correct += int(((predicted == EOS)&(ids == EOS)).sum())
        for item in predicted.tolist():
            try:
                decode(item)
                valid_text += 1
            except ValueError:
                pass
        for i, c in enumerate(codes):
            used[i].update(c.flatten().tolist())
    n = len(rows)
    if not n:
        raise ValueError('No evaluation rows')
    return {'examples':n, 'nonpad_symbol_accuracy':correct/total, 'exact_chunk_rate':exact/n,
            'eos_accuracy':eos_correct/n, 'valid_utf8_with_eos_rate':valid_text/n,
            'used_codes_per_level':list(map(len,used)),
            'passed':correct/total >= .999 and exact/n >= .95 and eos_correct/n >= .99,
            'thresholds':{'nonpad_symbol_accuracy':.999,'exact_chunk_rate':.95,'eos_accuracy':.99},
            'scope':'codec reconstruction only; not generated-code correctness'}


def rng_state():
    return {'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python':random.getstate()}


def set_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])


def train_codec(a):
    device = a.devices.split(',')[0]
    limits = admission([device], 4)
    rows, manifest = load_data(a.data)
    train = [r for r in rows if r['split']=='train']
    val = [r for r in rows if r['split']=='validation']
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=a.resume)
    cfg = CodecConfig(length=manifest['length'], levels=tuple(map(int,a.levels.split(','))), width=a.width,
                      layers=getattr(a,'layers',2),codes=getattr(a,'codes',512),
                      downsample=getattr(a,'downsample','linear'),coarse_loss_weight=getattr(a,'coarse_loss_weight',0.))
    torch.manual_seed(a.seed); random.seed(a.seed)
    codec = Codec(cfg).to(device)
    initialization = None
    if getattr(a,'init_checkpoint',None):
        initialization = initialize_codec(codec,a.init_checkpoint)
    optimizer = optimizer_for(codec,a)
    identity = {'data_sha256':manifest['examples_sha256'], 'config':dataclasses.asdict(cfg),
                'lr':a.lr, 'batch':a.batch, 'seed':a.seed, 'stage':'codec', 'format':1}
    record_optimizer(identity,a)
    if initialization is not None:identity['initialization']=initialization
    begin = 0; best = -1.; history = []
    if a.resume:
        ck = torch.load(output/'latest.pt', weights_only=True, map_location='cpu')
        stored_identity=dict(ck['identity'])
        stored_identity['config']=dataclasses.asdict(CodecConfig(**stored_identity['config']))
        if stored_identity != identity:
            raise ValueError('Resume identity differs')
        codec.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);set_rng(ck['rng'])
        begin=ck['step'];best=ck['best'];history=ck['history']
    if a.steps <= begin:
        raise ValueError('Target step count must exceed the resumed checkpoint step')
    ids_all = torch.tensor([encode(r['target'],cfg.length) for r in train])
    write_json(output/'admission.json', limits)
    write_json(output/'identity.json', identity)
    started = time.monotonic()
    for step in range(begin, a.steps):
        headroom([device])
        codec.train()
        indices = torch.randint(len(train), (a.batch,))
        ids = ids_all[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = codec(ids)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite codec loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step+1)%a.eval_every==0 or step+1==a.steps:
            metrics = reconstruction(codec, val, device)
            record = {'step':step+1, 'loss':float(loss.detach()), 'gradient_norm':float(norm),
                      'elapsed_seconds':time.monotonic()-started, 'validation':metrics}
            history.append(record)
            print(json.dumps(record), flush=True)
            score = metrics['nonpad_symbol_accuracy']
            if score > best:
                best = score
                save(output/'best.pt', {'config':codec.metadata(),'model':cpu_tree(codec.state_dict()),
                    'identity':identity,'step':step+1,'validation':metrics})
                write_json(output/'gate.json', dict(metrics, checkpoint_sha256=digest(output/'best.pt'),
                    data_sha256=manifest['examples_sha256'], status='codec-only'))
            save(output/'latest.pt', {'identity':identity, 'model':cpu_tree(codec.state_dict()),
                'optimizer':cpu_tree(optimizer.state_dict()),'step':step+1,'best':best,'history':history,'rng':rng_state()})
            write_json(output/'training.json', {'stage':'codec','complete':step+1==a.steps,'history':history,
                'quality_ready':False,'meaning':'finite codec run; generator not trained by this command'})


def train_generator(a):
    from .backbone import Source, OxBackbone, ConceptGenerator, SOURCE_REVISION
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    devices = a.devices.split(',')
    limits = admission(devices)
    rows, manifest = load_data(a.data)
    codec_path = Path(a.codec)
    codec_sha = digest(codec_path)
    ck = torch.load(codec_path, weights_only=True, map_location='cpu')
    if digest(codec_path) != codec_sha:
        raise RuntimeError('Codec changed during loading; use an immutable checkpoint copy')
    if ck['identity']['data_sha256'] != manifest['examples_sha256']:
        raise ValueError('Codec and training data differ; prepare an explicitly validated new stage')
    if not ck['validation']['passed'] and not a.smoke:
        raise RuntimeError('Codec reconstruction gate failed; only --smoke structural checks are allowed')
    curriculum=bool(getattr(a,'exact_target_curriculum',False))
    if not a.smoke and not curriculum:
        require_complete_gate(getattr(a,'complete_gate',None),codec_sha,manifest)
    if curriculum:
        if not getattr(a,'complete_data',None) or not a.complete_gate:
            raise ValueError('Exact-target curriculum requires complete conditioning and a measured reconstruction report')
        broad_report=json.loads(Path(a.complete_gate).read_text())
        if broad_report.get('checkpoint_sha256')!=codec_sha or broad_report.get('corpus_sha256')!=manifest.get('complete_corpus_sha256'):
            raise ValueError('Curriculum reconstruction report identity differs')
    if a.smoke and a.steps > 3:
        raise ValueError('Smoke runs limited to three updates; no quality claim allowed')
    output = Path(a.output);output.mkdir(parents=True,exist_ok=a.resume)
    write_json(output/'admission.json', limits)
    torch.manual_seed(a.seed);random.seed(a.seed)
    codec = Codec(CodecConfig(**ck['config'])).to(devices[-1])
    codec.load_state_dict(ck['model']);codec.requires_grad_(False);codec.eval()
    source = Source(a.source)
    layer_split=list(map(int,a.layer_split.split(','))) if getattr(a,'layer_split',None) else None
    backbone = OxBackbone(source, devices, a.rank,layer_split=layer_split)
    if a.audit_reference:
        from .audit_backbone import audit
        report = audit(backbone, source, a.audit_reference)
        write_json(output/'backbone-audit.json', report)
        if not report['passed']:
            raise RuntimeError('AR backbone probe mismatch; inspect backbone-audit.json before adaptation')
    conditioned=bool(getattr(a,'complete_data',None))
    parallel=bool(getattr(a,'parallel_attention',False))
    model = ConceptGenerator(backbone, codec.config,parallel_attention=parallel,termination=conditioned,
                             concept_input_norm=getattr(a,'concept_input_norm',0.) or None)
    write_json(output/'transfer.json', {'identity':backbone.source_identity,'tensors':backbone.transfers,
        'all_32_layers_loaded':len(backbone.layers)==32,'quantization':'dequantized original Q4_K/Q6_K into frozen BF16',
        'untrained_parts':['residual adapters','latent projection','level and position embeddings','concept head']})
    headroom(devices)
    tokdir = snapshot_download('OrionLLM/OxCoder-9B',revision=SOURCE_REVISION,
        allow_patterns=['tokenizer*','config.json','chat_template.jinja'],local_dir=output.parent/'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokdir, trust_remote_code=False)
    prepared=None
    if conditioned:
        from .conditioning import index_responses,select_conditioned
        complete,complete_manifest=load_data(a.complete_data)
        teacher,teacher_manifest=load_data(a.teacher_data)
        if (complete_manifest['examples_sha256']!=manifest['complete_corpus_sha256']
                or teacher_manifest['examples_sha256']!=manifest['teacher_sha256']):
            raise ValueError('Conditioning sources differ from the codec corpus')
        prepared,selection=select_conditioned(rows,index_responses(complete,teacher),tokenizer,codec.config.length,a.prompt_tokens)
        if not any(r['split']=='train' for r,c in prepared) or not any(r['split']=='validation' for r,c in prepared):
            raise ValueError('No complete conditioning within budget for train/validation')
        expected_sources={r['source'] for r in rows if r['split']=='train'}
        if {r['source'] for r,c in prepared if r['split']=='train'}!=expected_sources:
            raise ValueError('Prompt budget excluded an entire training source')
        write_json(output/'conditioning.json',selection)
        if curriculum:
            prepared,curriculum_report=exact_training_curriculum(codec,prepared,devices[-1])
            write_json(output/'curriculum.json',curriculum_report)
            write_json(output/'broad-reconstruction.json',broad_report)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_for(model,a,foreach=False)
    identity = {'stage':'generator','format':1,'source_sha256':source.sha,'codec_sha256':codec_sha,
        'codec_step':ck['step'],
        'data_sha256':manifest['examples_sha256'],'rank':a.rank,'lr':a.lr,'seed':a.seed,
        'prompt_tokens':a.prompt_tokens,'smoke':a.smoke,'architecture':backbone.source_identity['architecture']}
    record_optimizer(identity,a)
    if conditioned:identity['conditioning']='full chat template, full previous chunks, explicit response termination'
    if parallel:identity['full_attention']='causal prefix; bidirectional target level; recurrent layers remain causal'
    if layer_split:identity['layer_split']=layer_split
    if curriculum:identity['admission']='exactly reconstructed training subset; broader reconstruction remains unqualified'
    if getattr(a,'concept_input_norm',0):identity['concept_input_norm']=a.concept_input_norm
    if getattr(a,'byte_loss',0):identity['decoded_byte_loss_weight']=a.byte_loss
    if getattr(a,'balanced_curriculum',False):identity['sampling']='uniform represented source/distillation/response-end strata, then uniform chunk'
    if getattr(a,'kernel_backend','reference')!='reference':
        import fla
        if not a.kernel_audit:raise ValueError('FLA requires a recorded passing operator/gradient audit')
        kernel_report=json.loads(Path(a.kernel_audit).read_text())
        if (not kernel_report.get('passed') or kernel_report.get('fla')!=fla.__version__
                or kernel_report.get('torch')!=str(torch.__version__)):
            raise ValueError('Kernel audit is failed or mismatched')
        identity['kernel_backend']={'name':'fla','version':fla.__version__,'torch':str(torch.__version__)}
        identity['kernel_audit_sha256']=digest(a.kernel_audit)
    if getattr(a,'generator_init',None):
        initial_sha=digest(a.generator_init)
        initial=torch.load(a.generator_init,map_location='cpu',weights_only=True)
        if digest(a.generator_init)!=initial_sha:raise RuntimeError('Generator initialization changed while loading')
        for key in ('source_sha256','codec_sha256','data_sha256','rank','architecture','conditioning','full_attention'):
            if initial['identity'].get(key)!=identity.get(key):raise ValueError(f'Initialization differs: {key}')
        if set(initial['model'])!=set(model.trainable_state()):raise ValueError('Initialization parameter inventory differs')
        with torch.no_grad():
            for name,p in model.named_parameters():
                if p.requires_grad:p.copy_(initial['model'][name].to(p))
        identity['initialization']={'checkpoint_sha256':initial_sha,'source_step':initial['step'],
                                    'scope':'weights only; new optimizer, sampling and update count'}
        del initial
    begin=0;history=[]
    if a.resume:
        previous = torch.load(output/'latest.pt',map_location='cpu',weights_only=True)
        if identity != previous['identity']:
            raise ValueError('Resume identity differs')
        state = previous['model']
        if set(state) != set(model.trainable_state()):
            raise ValueError('Resume parameter inventory differs')
        with torch.no_grad():
            for name,p in model.named_parameters():
                if p.requires_grad:p.copy_(state[name].to(p))
        optimizer.load_state_dict(previous['optimizer']);set_rng(previous['rng'])
        begin=previous['step'];history=previous['history']
    if a.steps <= begin:
        raise ValueError('Target step count must exceed the resumed checkpoint step')
    train = [r for r in rows if r['split']=='train']
    if conditioned:train=[(r,c) for r,c in prepared if r['split']=='train']
    buckets=curriculum_buckets(train) if getattr(a,'balanced_curriculum',False) else None
    if buckets:
        write_json(output/'sampling.json',{'strata':[{'source':b[0][0]['source'],
            'distilled':bool(b[0][0].get('distilled',False)), 'final_chunk':b[0][1]['is_final_chunk'],
            'examples':len(b)} for b in buckets], 'quality_ready':False})
    model.train()
    evaluations=json.loads((output/'generation-evaluations.json').read_text()) if a.resume and (output/'generation-evaluations.json').exists() else []
    if conditioned:
        if not any(e['step']==begin for e in evaluations):
            evaluations.append({'step':begin,**evaluate_candidate(model,codec,prepared,devices[-1])})
        write_json(output/'generation-evaluations.json',evaluations)
    for step in range(begin, a.steps):
        headroom(devices)
        if conditioned:row,condition=random.choice(random.choice(buckets) if buckets else train)
        else:row = train[step % len(train)]
        target = torch.tensor([encode(row['target'],codec.config.length)],device=devices[-1])
        codes, inputs = codec.targets(target)
        if conditioned:token_ids=condition['ids']
        else:
            token_ids = tokenizer.encode(row['prompt'],add_special_tokens=False)
            token_ids = token_ids[-a.prompt_tokens:]
        if not token_ids:raise ValueError('Empty conditioning prompt')
        prompt = torch.tensor([token_ids],dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        losses=[];started=time.monotonic()
        # One level per smoke update; full training accumulates all levels equally.
        smoke_levels = [0, len(codes)//2, len(codes)-1]
        levels = [smoke_levels[step % 3]] if a.smoke else list(range(len(codes)))
        for level in levels:
            with_stop=conditioned and level==len(codes)-1
            if with_stop:logits,stop=model(inputs[level],prompt,level,return_stop=True)
            else:logits = model(inputs[level],prompt,level)
            loss = F.cross_entropy(logits.flatten(0,1),codes[level].to(logits.device).flatten())/len(levels)
            if with_stop:loss=loss+.1*F.binary_cross_entropy_with_logits(stop,torch.full_like(stop,float(condition['is_final_chunk'])))
            if level==len(codes)-1 and getattr(a,'byte_loss',0):
                loss=loss+a.byte_loss*decoded_byte_loss(codec,logits,inputs[level],target)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite concept loss')
            loss.backward();losses.append(float(loss.detach()))
        adapter_norms=[]
        for adapter in backbone.adapters:
            grad=adapter.up.weight.grad
            if grad is None or not torch.isfinite(grad).all() or grad.norm()==0:
                raise RuntimeError('Loss did not reach every layer adapter')
            adapter_norms.append(float(grad.float().norm()))
        for p in parameters:
            if p.grad is not None and not torch.isfinite(p.grad).all():raise RuntimeError('Nonfinite gradient')
        # Clip separately per device to avoid implicit cross-device foreach operations.
        for d in devices:
            torch.nn.utils.clip_grad_norm_([p for p in parameters if p.device==torch.device(d)],1.,foreach=False,error_if_nonfinite=True)
        before = model.head.weight.detach().clone()
        optimizer.step()
        if torch.equal(before,model.head.weight):raise RuntimeError('No parameter update')
        record={'step':step+1,'loss':sum(losses),'levels':levels,'seconds':time.monotonic()-started,
                'adapter_gradient_norms':adapter_norms,'training_example':row['id'],
                'peak_allocated_bytes':{d:torch.cuda.max_memory_allocated(d) for d in devices if d.startswith('cuda')}}
        history.append(record);print(json.dumps(record),flush=True)
        if (step+1)%getattr(a,'save_every',1)==0 or step+1==a.steps:
            save(output/'latest.pt',{'identity':identity,'model':model.trainable_state(),
                'optimizer':cpu_tree(optimizer.state_dict()),'rng':rng_state(),'step':step+1,'history':history})
        write_json(output/'training.json',{'identity':identity,'complete':step+1==a.steps,'history':history,
            'quality_ready':False,'status':'structural-smoke-only' if a.smoke else 'trained-candidate-needs-heldout-evaluation'})
        if conditioned and ((step+1)%a.eval_every==0 or step+1==a.steps):
            evaluations.append({'step':step+1,**evaluate_candidate(model,codec,prepared,devices[-1])})
            write_json(output/'generation-evaluations.json',evaluations)
    # Verify serialization by reloading trainable tensors and comparing a fixed forward.
    model.eval()
    with torch.no_grad():
        expected=model(inputs[0],prompt,0)
        saved=torch.load(output/'latest.pt',weights_only=True,map_location='cpu')['model']
        model.head.bias.add_(1.)
        if torch.equal(expected, model(inputs[0],prompt,0)):
            raise RuntimeError('Reload check perturbation did not change output')
        for name,p in model.named_parameters():
            if p.requires_grad:p.copy_(saved[name].to(p))
        actual=model(inputs[0],prompt,0)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        generated, _=model.generate(codec,prompt)
    try:
        decode(generated[0].tolist())
        valid_generated_text = True
    except ValueError:
        valid_generated_text = False
    write_json(output/'structural-validation.json',{'checkpoint_reload_exact':True,
        'all_32_adapters_receive_gradients':True,'generated_byte_ids':generated[0].tolist(),
        'valid_generated_utf8_with_eos':valid_generated_text,
        'generation_passes':len(codec.config.levels),'quality_ready':False,
        'scope':'structural conversion checks; generated output is not a validated coding model'})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['codec','generator'])
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--devices',default='cuda:1');p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--levels',default='1,4,16,64,128');p.add_argument('--width',type=int,default=128)
    p.add_argument('--layers',type=int,default=2,help='Codec attention blocks; zero tests tokenwise reconstruction')
    p.add_argument('--codes',type=int,default=512,help='Entries per residual codebook')
    p.add_argument('--downsample',choices=['linear','area'],default='linear',help='Codec downsampling; changing this requires a new codec and generator targets')
    p.add_argument('--coarse-loss-weight',type=float,default=0.,help='Auxiliary reconstruction from a randomly retained coarse prefix, in addition to full reconstruction')
    p.add_argument('--optimizer',choices=['adamw','muon'],default='adamw')
    p.add_argument('--muon-lr',type=float,default=None,help='Defaults to --lr using AdamW RMS-matched Muon scaling')
    p.add_argument('--source');p.add_argument('--codec');p.add_argument('--rank',type=int,default=8)
    p.add_argument('--audit-reference', help='llama.cpp token fixture to check before any adaptation')
    p.add_argument('--prompt-tokens',type=int,default=64);p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--init-checkpoint',help='Codec weights-only initialization; use the same file when resuming this new experiment')
    p.add_argument('--complete-gate',help='Matching passing whole-response codec report; required outside structural smoke tests')
    p.add_argument('--complete-data',help='Complete corpus for faithful generator conditioning and termination targets')
    p.add_argument('--teacher-data',help='Matching teacher corpus; required with --complete-data')
    p.add_argument('--parallel-attention',action='store_true',help='Full-attention blocks see all positions in the current concept level')
    p.add_argument('--layer-split',help='Explicit number of layers per device, e.g. 12,20')
    p.add_argument('--exact-target-curriculum',action='store_true',help='Train only individually verified exact targets; does not pass release validation')
    p.add_argument('--save-every',type=int,default=1,help='Generator checkpoint interval')
    p.add_argument('--generator-init',help='Weights-only generator initialization for an explicitly new stage')
    p.add_argument('--balanced-curriculum',action='store_true',help='Balance source, distillation and response-ending strata')
    p.add_argument('--byte-loss',type=float,default=0.,help='Auxiliary decoded-byte CE weight on final level')
    p.add_argument('--kernel-backend',choices=['reference','fla'],default='reference',help='Record an explicitly selected isolated training backend')
    p.add_argument('--kernel-audit',help='Passing operator/gradient comparison for the selected FLA backend')
    p.add_argument('--concept-input-norm',type=float,default=0.,help='Normalize concept vectors to this L2 norm; zero preserves legacy behavior')
    a=p.parse_args()
    if a.steps<1 or a.eval_every<1 or a.batch<1 or a.prompt_tokens<1: p.error('Counts must be positive')
    if a.save_every<1:p.error('--save-every must be positive')
    if a.byte_loss<0:p.error('--byte-loss must be nonnegative')
    if a.concept_input_norm<0:p.error('--concept-input-norm must be nonnegative')
    if not 0<=a.coarse_loss_weight<=1:p.error('--coarse-loss-weight must be in [0,1]')
    if a.stage!='codec' and (a.downsample!='linear' or a.coarse_loss_weight):p.error('Codec architecture options apply only to codec training; generators read the frozen checkpoint configuration')
    if a.balanced_curriculum and not a.complete_data:p.error('Balanced curriculum requires complete conditioning')
    if a.stage!='generator' and (a.generator_init or a.balanced_curriculum or a.byte_loss):p.error('Generator-only option')
    if a.stage=='generator':
        import importlib.util
        has_fla=importlib.util.find_spec('fla') is not None
        if has_fla!=(a.kernel_backend=='fla'):p.error('FLA availability differs from --kernel-backend; use an isolated PYTHONPATH')
    torch.set_num_threads(2)
    if a.stage=='generator' and (not a.source or not a.codec): p.error('Generator requires --source and --codec')
    if a.stage=='generator' and a.init_checkpoint:p.error('--init-checkpoint applies only to the codec')
    if bool(a.complete_data)!=bool(a.teacher_data):p.error('--complete-data and --teacher-data must be supplied together')
    (train_codec if a.stage=='codec' else train_generator)(a)


if __name__=='__main__':main()
