# watch_v6.py — env_v6(골대 조준) 시각화. watch_v5a 기반, obs 50 자동.
import argparse, time, pickle
import numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import PPO
from env_v6 import OP3KickEnvV6

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=r"runs\op3_kick_v6_s0.zip")
ap.add_argument("--vecnorm", default=r"runs\vecnorm_v6_s0.pkl")
ap.add_argument("--max_steps", type=int, default=300)
ap.add_argument("--hold", type=float, default=3.0)
ap.add_argument("--goal_randomize", action="store_true")
a = ap.parse_args()

with open(a.vecnorm, "rb") as f:
    vn = pickle.load(f)
obs_rms, eps, clip = vn.obs_rms, vn.epsilon, vn.clip_obs


def norm_obs(o):
    o = np.asarray(o, dtype=np.float32)
    return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + eps), -clip, clip).astype(np.float32)


model = PPO.load(a.model, device="cpu")
env = OP3KickEnvV6(kick_window=None, max_steps=a.max_steps,
                   goal_as_target=True, goal_randomize=a.goal_randomize)
mj_model, mj_data = env.model, env.data
print("model :", a.model, " — 자동 반복, Ctrl+C로 종료. obs_dim:", env.observation_space.shape)
with mujoco.viewer.launch_passive(mj_model, mj_data) as v:
    while v.is_running():
        o, info = env.reset(); env.reset_bonus()
        print("goal_pos=%s theta=%.2f" % (env.goal_pos, info["theta"]))
        for _ in range(a.max_steps):
            act, _ = model.predict(norm_obs(o), deterministic=True)
            o, _, te, tr, inf = env.step(act)
            v.sync(); time.sleep(0.03)
            if not v.is_running() or te or tr:
                break
        t0 = time.time()
        while time.time() - t0 < a.hold and v.is_running():
            v.sync(); time.sleep(0.03)
