"""Reconstruct complete prompt history and chunk termination without truncation."""
from .data import chunks
from .distill import normalize


def index_responses(corpus,teacher):
    result={}
    groups={r['group']:r['split'] for r in corpus}
    for r in teacher:
        if r['split']!='train' or groups.get(r['group'])!='train':
            raise ValueError('Teacher response crossed conversation split')
    for r in corpus+teacher:
        if r['id'] in result:raise ValueError('Duplicate response ID')
        result[r['id']]=r
    return result


def condition_chunk(row,responses,tokenizer,length,token_limit,memo=None):
    base,separator,index=row['id'].rpartition(':byte-')
    if not separator or base not in responses:raise ValueError('Unknown chunk provenance')
    original=responses[base];i=int(index)
    memo={} if memo is None else memo
    if base not in memo:
        parts=list(chunks(original['target'],length-1))
        messages=normalize(original['messages'])
        opening=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        opening_ids=tokenizer.encode(opening,add_special_tokens=False)
        # Appending text cannot merge BPE tokens across a recognized special-token boundary.
        special=set(getattr(tokenizer,'all_special_ids',[]))
        stable_prefix=max((j+1 for j,t in enumerate(opening_ids) if t in special),default=0)
        memo[base]=(parts,opening,stable_prefix)
    parts,opening,stable_prefix=memo[base]
    if (i<0 or i>=len(parts) or parts[i]!=row['target']
            or row['group']!=original['group'] or row['split']!=original['split']):
        raise ValueError('Chunk differs from original response')
    if stable_prefix>token_limit:return None
    prompt=opening+''.join(parts[:i])
    ids=tokenizer.encode(prompt,add_special_tokens=False)
    if not ids:raise ValueError('Empty conditioning')
    # Oversized examples are explicitly skipped, never changed into another task.
    if len(ids)>token_limit:return None
    return {'ids':ids,'is_final_chunk':i==len(parts)-1,'response_id':base,
            'chunk_index':i,'prompt_tokens':len(ids)}


def select_conditioned(rows,responses,tokenizer,length,token_limit):
    accepted=[];counts={};memo={}
    for row in rows:
        key=row['split']+'|'+row['source'];stats=counts.setdefault(key,{'kept':0,'over_limit':0})
        c=condition_chunk(row,responses,tokenizer,length,token_limit,memo)
        if c is None:stats['over_limit']+=1
        else:stats['kept']+=1;accepted.append((row,c))
    return accepted,{'counts':counts,'token_limit':token_limit,'truncation':False,
                     'format':'source chat template plus complete previous assistant chunks'}
