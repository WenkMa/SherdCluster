#!/usr/bin/env python3
"""Portable entry point. Paths are relative to the bundle root."""
import argparse,json,os,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('stage',choices=['train','extract','cluster','evaluate','reproduce'])
 p.add_argument('--device',default='cuda:0');p.add_argument('--config',type=Path)
 a,extra=p.parse_known_args();root=Path(__file__).resolve().parent;os.chdir(root)
 def execute(script,args):
  cmd=[sys.executable,script,*args];print('Running:', ' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
 scripts={'train':'train_byol_global_sides.py','extract':'extract_byol_global_sides.py','cluster':'cluster.py'}
 def stage(name, overrides=None):
  c=json.loads((a.config or Path('configs')/(name+'.json')).read_text());c.update(overrides or {});args=[]
  for key,value in c.items():
   if value is not None:args.extend(['--'+key.replace('_','-'),str(value)])
  execute(scripts[name],[*args,'--device',a.device,*extra])
 if a.stage in scripts:stage(a.stage)
 elif a.stage=='evaluate':
  execute('evaluate.py',['--features','outputs/features/exterior.npz','outputs/features/interior.npz','outputs/features/both_concat.npz',
                        '--output','outputs/evaluation','--device',a.device,*extra])
 else:
  if extra or a.config:p.error('reproduce uses the bundled reference configuration; no extra overrides')
  stage('cluster', {'features':'data/features/both_concat.npz'})
  execute('evaluate.py',['--features','data/features/exterior.npz','data/features/interior.npz','data/features/both_concat.npz',
                        '--assignments','outputs/clusters/assignments.csv','--output','outputs/evaluation','--device',a.device])
if __name__=='__main__':main()
