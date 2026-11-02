"""Fetch only reviewed NCM source and codec artifacts, never the large generator."""
import argparse
import shutil
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download
from .backbone import digest
from .official_codec import UPSTREAM_COMMIT, HF_REVISION, ARTIFACTS
from .train import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',default='results/ncm-upstream');p.add_argument('--output',default='results/ncm-official')
    a=p.parse_args();upstream=Path(a.upstream)
    if not upstream.exists():
        upstream.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','https://github.com/HazyResearch/ncm.git',str(upstream)],check=True)
        subprocess.run(['git','checkout','--detach',UPSTREAM_COMMIT],cwd=upstream,check=True)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()!=UPSTREAM_COMMIT:
        raise ValueError('Existing upstream checkout has a different revision; not modifying it')
    for name,files in ARTIFACTS.items():
        dest=Path(a.output)/name;dest.mkdir(parents=True,exist_ok=True)
        for filename,expected in files.items():
            target=dest/filename
            if target.exists():
                if digest(target)!=expected:raise ValueError(f'Existing file mismatch: {target}')
            else:
                source=hf_hub_download('hazyresearch/ncm-checkpoints',f'{name}/{filename}',revision=HF_REVISION)
                if digest(source)!=expected:raise ValueError(f'Download hash mismatch: {filename}')
                shutil.copyfile(source,target)
        write_json(dest/'download.json',{'revision':HF_REVISION,'sha256':files})


if __name__=='__main__':main()
