import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.net import UniChessNet,NetConfig
from model import train_iteration as old
from model import train_iteration_fast as fast
torch.set_num_threads(4)
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(111)
m=UniChessNet(NetConfig(blocks=2,filters=16)).cuda().eval()
x=torch.randn(8,19,8,8,device='cuda')
with torch.no_grad():
    before=m(x)
    m.to(memory_format=torch.channels_last)
    after=m(x.to(memory_format=torch.channels_last))
    for a,b in zip(before,after):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
class Logits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p=torch.nn.Parameter(torch.randn(8,4096,device='cuda'))
        self.pr=torch.nn.Parameter(torch.randn(8,4,device='cuda'))
        self.w=torch.nn.Parameter(torch.randn(8,3,device='cuda'))
    def forward(self,x):return self.p,self.pr,self.w
for mode in ['mixed','no_policy','no_promotion']:
    p=torch.softmax(torch.randn(8,4096,device='cuda'),1)
    p[0]=0
    pr=torch.tensor([-100,0,1,2,3,-100,1,2],device='cuda')
    w=torch.softmax(torch.randn(8,3,device='cuda'),1)
    if mode=='no_policy':p.zero_()
    if mode=='no_promotion':pr.fill_(-100)
    test=Logits();batch=[x,p,pr,w]
    a=old.loss_for(test,batch)[0];a.backward()
    grads=[v.grad.clone() for v in test.parameters()];test.zero_grad(set_to_none=True)
    b=fast.loss_for(test,batch)[0];b.backward()
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    for g,v in zip(grads,test.parameters()):torch.testing.assert_close(g,v.grad,atol=2e-6,rtol=2e-5)
print('PASS: channels-last preserves outputs and loss/gradients match for mixed/missing labels',flush=True)
