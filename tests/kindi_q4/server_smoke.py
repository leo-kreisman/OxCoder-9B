"""Real HTTP/cache regression. Run under the same resource guard as inference."""
import json,os,subprocess,sys,time,urllib.request
from pathlib import Path
pack,report=sys.argv[1:];port=18193
state=str(Path(report).with_suffix('.state'))
env=dict(os.environ,KINDI_PARK_MAX='0',KINDI_CHECKPOINT_BUDGET_MB='192',KINDI_CHECKPOINT_EVERY='16',KINDI_PREFIX_MIN='16',KINDI_PREFIX_MAX='1')
server=subprocess.Popen([sys.executable,'-m','kindi','--pack',pack,'--ctx','512','--state-dir',state,'serve','--port',str(port)],env=env)
def request(path,body=None):
 data=None if body is None else json.dumps(body).encode()
 req=urllib.request.Request('http://127.0.0.1:%d%s'%(port,path),data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=90) as resp:return json.load(resp)
try:
 for _ in range(100):
  if server.poll() is not None:raise RuntimeError('Server exited')
  try:health=request('/health');break
  except OSError:time.sleep(.1)
 else:raise TimeoutError('Server did not start')
 body={'model':'oxcoder9b-q4-kindi','messages':[{'role':'user','content':'Write a Python function that returns the square of a number. Code only.'}],'temperature':0,'max_tokens':32}
 first=request('/v1/chat/completions',body);second=request('/v1/chat/completions',body)
 assert first['choices'][0]['message']['content']==second['choices'][0]['message']['content']
 assert second['usage']['prompt_tokens_details']['cached_tokens']>0
 code=first['choices'][0]['message']['content'].split('```python\n')[1].split('```')[0]
 # Exact expected function: do not execute arbitrary model-generated code.
 assert code.strip()=='def square(n):\n    return n ** 2'
 streaming=dict(body,stream=True,stream_options={'include_usage':True})
 req=urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions'%port,data=json.dumps(streaming).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=90) as resp:sse=resp.read().decode()
 assert 'data: [DONE]' in sse
 events=[json.loads(line[6:]) for line in sse.splitlines() if line.startswith('data: {')]
 streamed=''.join(e['choices'][0]['delta'].get('content','') for e in events if e.get('choices'))
 assert streamed==first['choices'][0]['message']['content']
 server.terminate();server.wait(timeout=10)
 server=subprocess.Popen([sys.executable,'-m','kindi','--pack',pack,'--ctx','512','--state-dir',state,'serve','--port',str(port)],env=env)
 for _ in range(100):
  try:request('/health');break
  except OSError:time.sleep(.1)
 restored=request('/v1/chat/completions',body)
 assert restored['usage']['prompt_tokens_details']['cached_tokens']>0
 assert restored['choices'][0]['message']['content']==first['choices'][0]['message']['content']
 Path(report).write_text(json.dumps(dict(health=health,first=first,second=second,streaming_pass=True,after_restart=restored),indent=2)+'\n')
 print('HTTP output and prefix cache PASS',flush=True)
finally:
 server.terminate()
 try:server.wait(timeout=10)
 except subprocess.TimeoutExpired:server.kill();server.wait()
