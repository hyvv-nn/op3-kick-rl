# watch_v5b.py — v5-B(마스킹+물리고정). 자동 반복 + 긴 표시.
import argparse, time, pickle
import numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import PPO
from env_v5b import OP3KickEnvV5B

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=r"runs\op3_kick_v5b_s0.zip")
ap.add_argument("--vecnorm", default=r"runs\vecnorm_v5b_s0.pkl")
ap.add_argument("--max_steps", type=int, default=300)
ap.add_argument("--hold", type=float, default=3.0)
a = ap.parse_args()

with open(a.vecnorm, "rb") as f:
    vn = pickle.load(f)
obs_rms, eps, clip = vn.obs_rms, vn.epsilon, vn.clip_obs


def norm_obs(o):
    o = np.asarray(o, dtype=np.float32)
    return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + eps), -clip, clip).astype(np.float32)


model = PPO.load(a.model, device="cpu")
env = OP3KickEnvV5B(kick_window=None, max_steps=a.max_steps, full_action=False)   # 마스킹+물리고정
mj_model, mj_data = env.model, env.data

print("model :", a.model, " — 자동 반복, Ctrl+C로 종료")
print("act_dim:", env.act_dim, " kick_idx:", env.kick_idx)
with mujoco.viewer.launch_passive(mj_model, mj_data) as v:
    while v.is_running():
        o, info = env.reset(); env.reset_bonus()
        print("target theta=%.2f D=%.2f" % (info["theta"], info["D"]))
        for _ in range(a.max_steps):
            act, _ = model.predict(norm_obs(o), deterministic=True)
            o, _, te, tr, inf = env.step(act)
            v.sync(); time.sleep(0.03)
            if not v.is_running():
                break
            if te or tr:
                break
        t0 = time.time()
        while time.time() - t0 < a.hold and v.is_running():
            v.sync(); time.sleep(0.03)