import os
os.makedirs('/home/claude/bench/model', exist_ok=True)
import numpy as np, torch, time, os, json, sys
sys.path.insert(0,'/home/claude/work')
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
from holographic.io_and_interop.holographic_unicron import save_safetensors
torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 4)
text=""
for n in ('dict','docs','code'):
    text += open('/home/claude/bench/%s.txt'%n, encoding='utf-8', errors='ignore').read()[:600000] + "\n"
data=np.frombuffer(text.encode('utf-8','ignore'), dtype=np.uint8).astype(np.int64)
cfg=Qwen3NextConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
 num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
 linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
 linear_value_head_dim=32, linear_conv_kernel_dim=4, full_attention_interval=4,
 num_experts=0, tie_word_embeddings=True, rms_norm_eps=1e-6)
model=Qwen3NextForCausalLM(cfg).float()
opt=torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
SEQ=128; BS=12; STEPS=int(sys.argv[1]) if len(sys.argv)>1 else 400
split=int(0.95*len(data)); train, val = data[:split], data[split:]
def batch(src, rng):
    i=rng.integers(0, len(src)-SEQ-1, size=BS)
    return (torch.tensor(np.stack([src[j:j+SEQ] for j in i])),
            torch.tensor(np.stack([src[j+1:j+SEQ+1] for j in i])))
rng=np.random.default_rng(0); t0=time.time()
print('corpus %d bytes | params %.2fM | %d steps' % (len(data), sum(p.numel() for p in model.parameters())/1e6, STEPS), flush=True)
for step in range(STEPS):
    x,y=batch(train,rng)
    loss=torch.nn.functional.cross_entropy(model(x).logits.reshape(-1,256), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step%100==0 and step>0:
        model.eval()
        save_safetensors('/home/claude/bench/model/model.safetensors',
          {k: np.ascontiguousarray(v.detach().numpy().astype(np.float32)) for k,v in model.state_dict().items()})
        json.dump(cfg.to_dict(), open('/home/claude/bench/model/config.json','w'), default=str)
        model.train()
    if step%50==0 or step==STEPS-1:
        model.eval()
        with torch.no_grad():
            xv,yv=batch(val,np.random.default_rng(7))
            vl=float(torch.nn.functional.cross_entropy(model(xv).logits.reshape(-1,256), yv.reshape(-1)))
        model.train()
        print('step %4d | train %.3f | val %.3f | val ppl %.1f | %.0fs' % (step, float(loss), vl, np.exp(vl), time.time()-t0), flush=True)
model.eval()
save_safetensors('/home/claude/bench/model/model.safetensors',
  {k: np.ascontiguousarray(v.detach().numpy().astype(np.float32)) for k,v in model.state_dict().items()})
json.dump(cfg.to_dict(), open('/home/claude/bench/model/config.json','w'), default=str)
with torch.no_grad():
    p=torch.tensor(np.frombuffer(b"the meaning of life is", dtype=np.uint8).astype(np.int64))[None]
    for _ in range(50):
        p=torch.cat([p, model(p).logits[0,-1].argmax().view(1,1)],1)
print('SAMPLE:', bytes(p[0].numpy().astype(np.uint8)).decode('utf-8','replace'), flush=True)
print('SAVED', flush=True)
