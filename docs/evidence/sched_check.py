import json,glob,torch
from diffusers import DDIMScheduler
p=glob.glob(r"D:/prompt_Project/models/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/*/scheduler/scheduler_config.json")[0]
s=DDIMScheduler.from_config(json.load(open(p)))
s.set_timesteps(40)
ts=s.timesteps.tolist()
ac=s.alphas_cumprod
print("first/last timesteps",ts[:4],ts[-3:],"n",len(ts))
print("set_alpha_to_one",s.config.set_alpha_to_one,"final_alpha_cumprod",float(s.final_alpha_cumprod),"ac[0]",float(ac[0]),"ac[ts0]",float(ac[ts[0]]))
for t in [6,12,18,24,30,36,40]:
    lvl = ts[t] if t<40 else None
    a_t = float(ac[lvl]) if lvl is not None else float(s.final_alpha_cumprod)
    j=t-8
    jj=max(j,0)
    a_j=float(ac[ts[jj]])
    print(f"t={t:2d} level_ts={lvl} abar_t={a_t:.5f} target_idx={jj} (requested {j}) ts={ts[jj]} abar_j={a_j:.5f} ratio={a_j/a_t:.4f} eff_jump={jj and t-jj or t-jj}")
# moment test
torch.manual_seed(0)
x0=torch.randn(200000)*0.5+1.0
for t in [12,30]:
    a_t=float(ac[ts[t]]); a_j=float(ac[ts[t-8]])
    xt=a_t**.5*x0+(1-a_t)**.5*torch.randn_like(x0)
    r=a_j/a_t
    xp=r**.5*xt+(1-r)**.5*torch.randn_like(x0)
    # target: sqrt(a_j)*x0 + sqrt(1-a_j) eps ; compare mean/var of (xp - sqrt(a_j)*x0)
    res=xp-a_j**.5*x0
    print(t,"residual var",float(res.var()),"expected",1-a_j,"mean",float(res.mean()))
