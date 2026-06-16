# watch_v4.py — v4(전신 자유+페널티). 자동 반복 + 긴 표시.
import argparse, time, pickle
import numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import PPO
from env_v4 import OP3KickEnvV4

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=r"runs\op3_kick_v4_s0.zip")
ap.add_argument("--vecnorm", default=r"runs\vecnorm_v4_s0.pkl")
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
env = OP3KickEnvV4(kick_window=None, max_steps=a.max_steps)   # eval=coast
mj_model, mj_data = env.model, env.data

print("model :", a.model, " — 자동 반복, Ctrl+C로 종료")
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