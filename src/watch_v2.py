# watch_v2.py — v2 직진 킥. 자동 반복(Enter 불필요) + 긴 표시.
import argparse, time, pickle
import numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import PPO
from op3_kick_env import OP3KickEnv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=r"runs\op3_kick_ppo_s0_g0.zip")
ap.add_argument("--vecnorm", default=r"runs\vecnorm_s0_g0.pkl")
ap.add_argument("--max_steps", type=int, default=200)
ap.add_argument("--hold", type=float, default=3.0)   # 킥 끝나고 멈춰서 보여주는 시간(초)
a = ap.parse_args()

with open(a.vecnorm, "rb") as f:
    vn = pickle.load(f)
obs_rms, eps, clip = vn.obs_rms, vn.epsilon, vn.clip_obs


def norm_obs(o):
    o = np.asarray(o, dtype=np.float32)
    return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + eps), -clip, clip).astype(np.float32)


model = PPO.load(a.model, device="cpu")
env = OP3KickEnv(max_steps=a.max_steps, goal_cond=False)
mj_model, mj_data = env.unwrapped.model, env.unwrapped.data

print("model :", a.model, " (goal_cond=False, 직진) — 자동 반복, Ctrl+C로 종료")
with mujoco.viewer.launch_passive(mj_model, mj_data) as v:   # 뷰어 1번만 열고 계속 유지
    while v.is_running():
        o, info = env.reset()
        for _ in range(a.max_steps):
            act, _ = model.predict(norm_obs(o), deterministic=True)
            o, _, te, tr, inf = env.step(act)
            v.sync(); time.sleep(0.03)
            if not v.is_running():
                break
            if te or tr:
                break
        t0 = time.time()                  # 킥 끝난 자세로 hold초 멈춰서 보여줌
        while time.time() - t0 < a.hold and v.is_running():
            v.sync(); time.sleep(0.03)