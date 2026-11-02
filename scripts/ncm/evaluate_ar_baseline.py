"""Measure original OxCoder on the development tasks; never use as training data."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

from .backbone import digest
from .coding_eval import TASKS,check_case,extract
from .train import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:18109');p.add_argument('--model',default='oxcoder-ar-baseline')
    p.add_argument('--output',required=True);p.add_argument('--source',required=True);p.add_argument('--server-binary',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    report={'source_sha256':digest(a.source),'server_binary_sha256':digest(a.server_binary),
        'temperature':0,'max_tokens':512,'reasoning':False,'tasks':[],
        'scope':'original AR source on the reused coding development diagnostic; never admitted to student training',
        'used_in_training':False,'quality_ready':False,'complete':False}
    for name,prompt,cases in TASKS:
        body={'model':a.model,'temperature':0,'max_tokens':512,
            'chat_template_kwargs':{'enable_thinking':False},
            'messages':[{'role':'system','content':'Return only Python code defining the requested function.'},
                        {'role':'user','content':prompt}]}
        request=urllib.request.Request(a.url+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
        started=time.monotonic()
        with urllib.request.urlopen(request,timeout=180) as response:r=json.load(response)
        choice=r['choices'][0];text=choice['message'].get('content','') or '';checks=[]
        if choice['finish_reason']=='stop':
            checks=[check_case(extract(text),name,args,want,preserve_arguments=name=='merge_intervals') for args,want in cases]
        row={'task':name,'text':text,'finish_reason':choice['finish_reason'],'usage':r.get('usage'),
             'seconds':time.monotonic()-started,'checks':checks,
             'passed':len(checks)==len(cases) and all(c['passed'] for c in checks)}
        report['tasks'].append(row);report['passed']=sum(x['passed'] for x in report['tasks'])
        report['evaluated']=len(report['tasks']);write_json(out/'report.json',report)
        print(json.dumps({'task':name,'passed':row['passed']}),flush=True)
    report['complete']=True;write_json(out/'report.json',report)


if __name__=='__main__':main()
