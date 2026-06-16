# watch_v3.py — v3 조준 킥(flailing 원점). 자동 반복 + 긴 표시.
import argparse, time, pickle
import numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import PPO
from op3_kick_env import OP3KickEnv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=r"runs\op3_kick_ppo_s0_g1.zip")
ap.add_argument("--vecnorm", default=r"runs\vecnorm_s0_g1.pkl")
ap.add_argument("--max_steps", type=int, default=200)
ap.add_argument("--hold", type=float, default=3.0)
a = ap.parse_args()

with open(a.vecnorm, "rb") as f:
    vn = pickle.load(f)
obs_rms, eps, clip = vn.obs_rms, vn.epsilon, vn.clip_obs


def norm_obs(o):
    o = np.asarray(o, dtype=np.float32)
    return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + eps), -clip, clip).astype(np.float32)


model = PPO.load(a.model, device="cpu")
env = OP3KickEnv(max_steps=a.max_steps, goal_cond=True)   # 조준 모드
mj_model, mj_data = env.unwrapped.model, env.unwrapped.data

print("model :", a.model, " (goal_cond=True, 조준) — 자동 반복, Ctrl+C로 종료")
with mujoco.viewer.launch_passive(mj_model, mj_data) as v:
    while v.is_running():
        o, info = env.reset()
        th = info.get("cmd_theta", None)
        print("target theta=%s" % ("%.2f" % th if th is not None else "?"))
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