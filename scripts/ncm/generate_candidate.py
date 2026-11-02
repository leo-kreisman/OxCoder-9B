"""Reference inference for a trained NCM candidate; never fall back to AR output."""
import argparse
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer
from .backbone import Source,OxBackbone,ConceptGenerator,digest
from .codec import Codec,CodecConfig,decode
from .train import admission,headroom,write_json
from .distill import normalize


def load_candidate(source_path,codec_path,checkpoint_path,devices):
    checkpoint_sha=digest(checkpoint_path)
    ck=torch.load(checkpoint_path,weights_only=True,map_location='cpu');identity=ck['identity']
    if identity.get('full_precision_reductions'):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    if digest(checkpoint_path)!=checkpoint_sha:raise RuntimeError('Checkpoint changed during load; use an immutable snapshot')
    if identity['codec_sha256']!=digest(codec_path):raise ValueError('Codec does not match generator')
    source=Source(source_path)
    if identity['source_sha256']!=source.sha:raise ValueError('Source does not match generator')
    c=torch.load(codec_path,weights_only=True,map_location='cpu')
    codec=Codec(CodecConfig(**c['config'])).to(devices[-1]);codec.load_state_dict(c['model']);codec.eval();codec.requires_grad_(False)
    backbone=OxBackbone(source,devices,identity['rank'],layer_split=identity.get('layer_split'))
    model=ConceptGenerator(backbone,codec.config,parallel_attention='full_attention' in identity,
                           termination='conditioning' in identity,concept_input_norm=identity.get('concept_input_norm'),
                           cached_prefix=bool(identity.get('cached_prefix',False)),
                           head_kind=identity.get('head_kind','linear'),codebooks=codec.books,
                           head_input_norm=bool(identity.get('head_input_norm',False)),
                           nearest_code=bool(identity.get('nearest_code',False)))
    if set(model.trainable_state())!=set(ck['model']):raise ValueError('Trainable checkpoint inventory differs')
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:p.copy_(ck['model'][name].to(p))
    model.eval();return model,codec,identity,checkpoint_sha


@torch.no_grad()
def generate(model,codec,tokenizer,messages,token_limit,max_chunks):
    opening=tokenizer.apply_chat_template(normalize(messages),tokenize=False,
                                          add_generation_prompt=True,enable_thinking=False)
    text='';steps=[];reason='chunk_limit';start=time.monotonic()
    for i in range(max_chunks):
        headroom([str(d) for d in model.backbone.devices])
        ids=tokenizer.encode(opening+text,add_special_tokens=False)
        if len(ids)>token_limit:reason='context_limit';break
        begin=time.monotonic()
        output,codes,stop=model.generate_chunk(codec,torch.tensor([ids],dtype=torch.long))
        try:piece=decode(output[0].tolist())
        except (ValueError,UnicodeError) as e:
            steps.append({'chunk':i,'seconds':time.monotonic()-begin,
                          'invalid_byte_ids':output[0].tolist(),'decode_error':str(e),
                          'stop_probability':float(stop[0])})
            reason='invalid_codec_output';break
        probability=float(stop[0]);text+=piece
        steps.append({'chunk':i,'seconds':time.monotonic()-begin,'bytes':len(piece.encode()),
                      'stop_probability':probability,'parallel_levels':len(codes)})
        if probability>=.5:reason='predicted_end';break
        if not piece:reason='empty_chunk';break
    return {'text':text,'stop_reason':reason,'chunks':steps,'seconds':time.monotonic()-start,
            'quality_ready':False,'autoregressive_fallback':False,
            'runtime':'Linux CUDA BF16 reference; not the 8GB deployment engine'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','codec','checkpoint','messages','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--devices',default='cuda:0,cuda:1');p.add_argument('--tokenizer',default='results/tokenizer')
    p.add_argument('--max-chunks',type=int,default=16);a=p.parse_args()
    torch.set_num_threads(2);devices=a.devices.split(',');admission(devices)
    model,codec,identity,checkpoint_sha=load_candidate(a.source,a.codec,a.checkpoint,devices)
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    result=generate(model,codec,tokenizer,json.loads(Path(a.messages).read_text()),identity['prompt_tokens'],a.max_chunks)
    result['checkpoint_sha256']=checkpoint_sha;write_json(a.output,result)


if __name__=='__main__':main()
