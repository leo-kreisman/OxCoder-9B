"""Small executable-code diagnostic with isolated execution and external grading.

These elementary tasks cannot establish professional coding quality by themselves.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import torch
from transformers import AutoTokenizer
from .generate_candidate import load_candidate,generate
from .train import admission,write_json

TASKS=[
 ('factorial','Implement factorial(n). Return n! for nonnegative integers, including 0. Raise ValueError for negative n.',
  [([0],1),([1],1),([6],720),([-1],{'raises':'ValueError'})]),
 ('unique','Implement unique(items), returning distinct items in original order. Items are hashable.',
  [([[]],[]),([[3,1,3,2,1]],[3,1,2]),([['b','b','a']],['b','a'])]),
 ('chunks','Implement chunks(items, n): split a list into lists of at most n items, preserving order. Raise ValueError if n <= 0.',
  [([[1,2,3,4,5],2],[[1,2],[3,4],[5]]),([[],3],[]),([[1],0],{'raises':'ValueError'})]),
 ('balanced','Implement balanced(s), returning whether parentheses, square brackets and braces are correctly nested. Ignore other characters.',
  [(['x(a[b]{c})'],True),(['([)]'],False),([''],True),([']'],False)]),
 ('gcd','Implement gcd(a,b), returning the nonnegative greatest common divisor of two integers; gcd(0,0) is 0.',
  [([54,24],6),([-12,18],6),([0,0],0),([0,-7],7)]),
 ('merge_intervals','Implement merge_intervals(intervals). Merge overlapping or touching [start,end] intervals and return them sorted, without mutating the input.',
  [([[[5,7],[1,3],[3,4]]],[[1,4],[5,7]]),([[]],[]),([[[1,8],[2,3]]],[[1,8]])]),
 ('flatten','Implement flatten(items), recursively flattening nested lists while preserving all non-list elements and order.',
  [([[1,[2,[],[3]],4]],[1,2,3,4]),([[]],[]),([['ab',[None]]],['ab',None])]),
 ('rotate','Implement rotate(items,k), returning a new list rotated right by k positions; negative k rotates left. Empty input returns [].',
  [([[1,2,3],1],[3,1,2]),([[1,2,3],-1],[2,3,1]),([[],10],[]),([[1,2],4],[1,2])]),
 ('word_counts','Implement word_counts(text), splitting on whitespace and counting each case-sensitive word in a dictionary.',
  [(['a b a'],{'a':2,'b':1}),([''],{}),(['A\tA\nb'],{'A':2,'b':1})]),
 ('transpose','Implement transpose(matrix) for a rectangular list of lists. Return [] for []. Raise ValueError if rows have different lengths.',
  [([[[1,2],[3,4]]],[[1,3],[2,4]]),([[]],[]),([[[1],[2,3]]],{'raises':'ValueError'})]),
 ('binary_search','Implement binary_search(items,target) on sorted integers. Return the leftmost matching index, or -1 if absent.',
  [([[1,2,2,4],2],1),([[],1],-1),([[1,3],2],-1),([[1],1],0)]),
 ('is_prime','Implement is_prime(n), returning whether integer n is prime.',
  [([2],True),([1],False),([-3],False),([49],False),([97],True)])]

WORKER="""
import json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(262144,262144))
resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
resource.setrlimit(resource.RLIMIT_NPROC,(1,1))
r=json.load(sys.stdin)
namespace={}
try:
 exec(compile(r['code'],'<candidate>','exec'),namespace)
 value=namespace[r['function']](*r['arguments'])
 result={'value':value,'arguments_after':r['arguments']}
except BaseException as e:
 result={'exception':type(e).__name__}
print(json.dumps(result))
"""


def extract(text):
    match=re.search(r'```(?:python)?\s*\n(.*?)```',text,re.S)
    return match.group(1) if match else text.strip()


def check_case(code,function,arguments,expected,preserve_arguments=False):
    command=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv',
             '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
             '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/tmp',
             '/usr/bin/python3','-I','-c',WORKER]
    with tempfile.TemporaryFile() as log:
        try:
            result=subprocess.run(command,input=json.dumps({'code':code,'function':function,'arguments':arguments}).encode(),
                                  stdout=log,stderr=log,timeout=5)
            log.seek(0);response=json.loads(log.read(262145))
            if result.returncode:return {'passed':False,'reason':'worker_exit','exit_code':result.returncode}
        except subprocess.TimeoutExpired:return {'passed':False,'reason':'timeout'}
        except (ValueError,UnicodeError):return {'passed':False,'reason':'invalid_worker_output'}
    wanted={'exception':expected['raises']} if isinstance(expected,dict) and 'raises' in expected else {'value':expected}
    unchanged=not preserve_arguments or response.get('arguments_after')==arguments
    observed={k:v for k,v in response.items() if k!='arguments_after'}
    return {'passed':observed==wanted and unchanged,'observed':response}


def evaluate_model(model,codec,tokenizer,token_limit,checkpoint_sha,output,max_chunks=8):
    results=[];started=time.monotonic()
    for name,prompt,cases in TASKS:
        generated=generate(model,codec,tokenizer,[{'role':'system','content':'Return only Python code defining the requested function.'},
                            {'role':'user','content':prompt}],token_limit,max_chunks)
        checks=[]
        if generated['stop_reason']=='predicted_end':
            code=extract(generated['text']);checks=[check_case(code,name,args,want,preserve_arguments=name=='merge_intervals') for args,want in cases]
        results.append({'task':name,'generation':generated,'checks':checks,
                        'passed':len(checks)==len(cases) and all(c['passed'] for c in checks)})
        report={'checkpoint_sha256':checkpoint_sha,'tasks':results,'passed':sum(r['passed'] for r in results),
                'evaluated':len(results),'seconds':time.monotonic()-started,'quality_ready':False,
                'scope':'12 elementary Python function tasks; not professional qualification or an 8GB measurement'}
        write_json(output,report);print(json.dumps({'task':name,'passed':results[-1]['passed']}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','codec','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--devices',default='cuda:0,cuda:1');p.add_argument('--tokenizer',default='results/tokenizer')
    p.add_argument('--max-chunks',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(2);devices=a.devices.split(',');admission(devices)
    model,codec,identity,sha=load_candidate(a.source,a.codec,a.checkpoint,devices)
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    evaluate_model(model,codec,tokenizer,identity['prompt_tokens'],sha,a.output,a.max_chunks)


if __name__=='__main__':main()
