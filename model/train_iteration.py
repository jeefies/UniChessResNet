"""46M supervised iteration with endgame/promotion oversampling and resumable checkpoints.

This is the supervised bootstrap phase, not a self-play training loop.
"""
import argparse
import json
import math
import time
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.record import RECORD_DTYPE
from model.dataset import decode_batch, decode_targets
from model.net import NetConfig, UniChessNet, count_params


def save_atomic(obj, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temp)
    temp.replace(path)


class Pool:
    def __init__(self, paths, cache):
        self.paths = paths
        self.maps = [np.memmap(p, dtype=RECORD_DTYPE, mode='r') for p in paths]
        self.offsets = np.cumsum([0] + [len(m) for m in self.maps])
        self.total = int(self.offsets[-1])
        fingerprint = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        manifest = cache.with_suffix('.json')
        expected = json.dumps(fingerprint)
        if cache.exists() and manifest.exists() and manifest.read_text() == expected:
            with np.load(cache) as d:
                self.endgame, self.promotion = d['endgame'], d['promotion']
        else:
            rng = np.random.default_rng(20260909)
            groups = [[], []]
            for shard, data in enumerate(self.maps):
                selected = [[], []]
                for start in range(0, len(data), 262144):
                    r = data[start:start+262144]
                    occ = np.asarray(r['occ_white'] | r['occ_black']).copy()
                    counts = np.unpackbits(occ.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
                    # White pawns on ranks 6/7; black pawns on ranks 2/3.
                    advanced = (((r['pawns'] & r['occ_white']) & np.uint64(0x00FFFF0000000000)) != 0) | (((r['pawns'] & r['occ_black']) & np.uint64(0x0000000000FFFF00)) != 0)
                    for i, mask in enumerate([counts <= 10, advanced]):
                        idx = np.flatnonzero(mask) + start + self.offsets[shard]
                        selected[i].append(idx)
                for i in range(2):
                    idx = np.concatenate(selected[i])
                    if len(idx) > 50000:
                        idx = rng.choice(idx, 50000, replace=False)
                    groups[i].append(idx.astype(np.int64))
                print('indexed',data.filename,flush=True)
            self.endgame, self.promotion = [np.concatenate(g) for g in groups]
            np.savez(cache, endgame=self.endgame, promotion=self.promotion)
            manifest.write_text(expected)
        if not len(self.endgame) or not len(self.promotion):
            raise ValueError('Specialty sample pool empty')
        print(json.dumps({'samples':self.total,'endgame_pool':len(self.endgame),'promotion_pool':len(self.promotion)}),flush=True)

    def sample(self, rng, n, mode='mixed'):
        if mode == 'mixed':
            a, b = n//2, n//4
            idx = np.concatenate([rng.integers(self.total,size=a), rng.choice(self.endgame,size=b), rng.choice(self.promotion,size=n-a-b)])
            rng.shuffle(idx)
        elif mode == 'general': idx = rng.integers(self.total,size=n)
        else: idx = rng.choice(getattr(self,mode),size=n)
        shards = np.searchsorted(self.offsets,idx,side='right')-1
        result = np.empty(n,dtype=RECORD_DTYPE)
        for s in np.unique(shards):
            sel = shards==s
            result[sel] = self.maps[s][idx[sel]-self.offsets[s]]
        x = decode_batch(result)
        p,pr,w = decode_targets(result)
        return [torch.from_numpy(t).to('cuda') for t in [x,p,pr,w]]


def loss_for(model, batch):
    x,p,pr,w=batch
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pl,prl,wl=model(x)
        has=p.sum(1)>0
        lp=-(p[has]*F.log_softmax(pl[has],dim=1)).sum(1).mean() if has.any() else pl.sum()*0
        lw=-(w*F.log_softmax(wl,dim=1)).sum(1).mean()
        eligible=pr!=-100
        lpr=F.cross_entropy(prl[eligible],pr[eligible]) if eligible.any() else prl.sum()*0
        loss=lp+lw+0.1*lpr
    return loss,lp,lw


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/shards_evals')
    ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=125000)
    ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--accum',type=int,default=8)
    ap.add_argument('--save-every',type=int,default=1000)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    torch.backends.cudnn.benchmark=True
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    paths=sorted(Path(args.data).glob('*.bin'))
    if len(paths)<8:raise ValueError('Need >=8 shards for held-out split')
    train=Pool(paths[:-4],out/'train_pools.npz')
    valid=Pool(paths[-4:],out/'valid_pools.npz')
    cfg=NetConfig(blocks=24,filters=320)
    model=UniChessNet(cfg).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    rng=np.random.default_rng(20260909);step=0;best=float('inf')
    if args.resume:
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert ck['cfg']==cfg.__dict__
        assert ck['args']['steps']==args.steps and ck['args']['batch']==args.batch and ck['args']['accum']==args.accum
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        step=ck['step'];best=ck['best'];rng.bit_generator.state=ck['rng']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    elif (out/'latest.pt').exists():raise ValueError('Checkpoint exists; use --resume')
    config={'args':vars(args),'cfg':cfg.__dict__,'params':count_params(model)['TOTAL'],'train_shards':[str(p) for p in paths[:-4]],'heldout_shards':[str(p) for p in paths[-4:]],'phase':'supervised bootstrap from random initialization','mix':{'general':0.5,'<=10 pieces':0.25,'advanced pawns':0.25}}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(config),flush=True)
    started=time.time();initial_step=step
    model.train()
    while step<args.steps:
        lr=3e-4*min(1,(step+1)/2000)*(0.05+0.95*0.5*(1+math.cos(math.pi*step/args.steps)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);totals=np.zeros(3)
        for _ in range(args.accum):
            loss,lp,lw=loss_for(model,train.sample(rng,args.batch))
            if not torch.isfinite(loss):raise FloatingPointError('non-finite loss')
            (loss/args.accum).backward()
            totals += [loss.item(),lp.item(),lw.item()]
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        if not torch.isfinite(norm):raise FloatingPointError('non-finite gradients')
        opt.step();step+=1
        if step==1 or step%10==0:
            record={'step':step,'loss':float(totals[0]/args.accum),'policy':float(totals[1]/args.accum),'value':float(totals[2]/args.accum),'lr':lr,'samples_per_sec':round((step-initial_step)*args.batch*args.accum/(time.time()-started),1),'gpu_peak_mib':round(torch.cuda.max_memory_allocated()/2**20)}
            print(json.dumps(record),flush=True)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if step == 10 or step%args.save_every==0 or step==args.steps:
            model.eval();scores={}
            with torch.no_grad():
                for mode in ['general','endgame','promotion']:
                    vrng=np.random.default_rng(991)
                    scores[mode]=float(np.mean([loss_for(model,valid.sample(vrng,args.batch,mode))[0].item() for _ in range(8)]))
            score=float(np.mean(list(scores.values())))
            improved=score<best;best=min(best,score)
            payload={'model':model.state_dict(),'cfg':cfg.__dict__,'step':step,'optimizer':opt.state_dict(),'best':best,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'args':vars(args)}
            save_atomic(payload,out/'latest.pt')
            if improved:save_atomic({'model':model.state_dict(),'cfg':cfg.__dict__,'step':step},out/'best.pt')
            print(json.dumps({'checkpoint':step,'validation':scores,'best':best}),flush=True)
            model.train()
    print('SUPERVISED_PHASE_COMPLETE',flush=True)

if __name__=='__main__':main()
