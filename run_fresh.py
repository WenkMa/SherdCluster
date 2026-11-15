#!/usr/bin/env python3
"""Fresh 15-epoch training -> full extraction -> clustering -> evaluation.

Only ImageNet initialization is loaded. No historical SherdCluster checkpoint,
feature bank or cluster result is read by this pipeline.
"""
import argparse,datetime,json,os,subprocess,sys,traceback
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--image-dir',type=Path,default=Path('data/images'))
 p.add_argument('--output',type=Path,default=Path('outputs/fresh_run'))
 p.add_argument('--device',default='cuda:0')
 p.add_argument('--workers',type=int,default=4)
 p.add_argument('--collection-ids',type=Path,help='Optional inventory including sherds with missing sides')
 a=p.parse_args();root=Path(__file__).resolve().parent;os.chdir(root)
 if not a.image_dir.is_dir():p.error('Place original paired surface images in --image-dir')
 a.output.mkdir(parents=True,exist_ok=False)
 config=json.loads(Path('configs/train.json').read_text())
 config.update(image_dir=str(a.image_dir),output=str(a.output/'train'),device=a.device,workers=a.workers)
 status=dict(state='running',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             pipeline='fresh training, full extraction, clustering and independent evaluation',
             initialization='ImageNet IMAGENET1K_V2 only; no previous sherd-domain weights',
             training_epochs=15,cosine_schedule_horizon=100,device=a.device,
             cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),completed_stages=[])
 def save():
  tmp=a.output/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(a.output/'status.json')
 def execute(stage,script,args):
  status['current_stage']=stage;status['stage_started_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
  command=[sys.executable,'-u',script,*args]
  print('Starting '+stage+': '+' '.join(['python',script,*args]),flush=True)
  with (a.output/(stage+'.log')).open('w') as f:
   subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
  status['completed_stages'].append(stage);save();print('Finished '+stage,flush=True)
 try:
  args=[]
  for key,val in config.items():args.extend(['--'+key.replace('_','-'),str(val)])
  execute('training','train_byol_global_sides.py',args)
  execute('extraction','extract_byol_global_sides.py',[
   '--checkpoint',str(a.output/'train/checkpoint_epoch15.pt'),'--image-dir',str(a.image_dir),
   '--output',str(a.output/'features'),'--device',a.device,'--batch-size','16','--workers',str(a.workers)])
  cluster_args=['--features',str(a.output/'features/both_concat.npz'),'--output',str(a.output/'clusters'),
                '--device',a.device,'--k','10','--eta','0.50','--rho','0.55']
  if a.collection_ids:cluster_args.extend(['--collection-ids',str(a.collection_ids)])
  execute('clustering','cluster.py',cluster_args)
  execute('evaluation','evaluate.py',[
   '--features',*[str(a.output/'features'/f'{name}.npz') for name in ['exterior','interior','both_concat']],
   '--assignments',str(a.output/'clusters/assignments.csv'),'--output',str(a.output/'evaluation'),
   '--device',a.device])
  status.update(state='complete',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 except BaseException as e:
  status.update(state='failed',error=str(e),finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
  save();raise
 save()
 print('Fresh pipeline complete; results in '+str(a.output),flush=True)
if __name__=='__main__':main()
