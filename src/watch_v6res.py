# watch_v6res.py — residual(env_v6res) 모델 시각화. watch_v6 기반 + 잔차 env.
#   잔차 모델(v6resb / v6res015 / v6res03)은 analytic baseline 위에 RL 잔차를 얹는 구조라
#   순수-RL 뷰어(watch_v6.py)로 띄우면 액션 매핑이 틀려 엉뚱한 동작이 나온다 → 본 뷰어 사용.
#   학습 분포 그대로(goal_as_target=False): 매 에피소드 무작위 명령 방향(theta)으로 찬다.
# 실행 예:
#   python watch_v6res.py --model runs\op3_kick_v6resb_s2.zip --vecnorm runs\vecnorm_v6resb_s2.pkl --res_scale 0.5 --hold 4
#   ※ --res_scale 은 반드시 그 모델을 *학습할 때 쓴 값*과 동일하게(resb=0.5, v6res015=0.15, v6res03=0.3).
import argparse, time, pickle
import numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import PPO
from env_v6res import OP3KickEnvV6Res

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=r"runs\op3_kick_v6resb_s2.zip")
ap.add_argument("--vecnorm", default=r"runs\vecnorm_v6resb_s2.pkl")
ap.add_argument("--res_scale", type=float, default=0.5, help="학습에 쓴 값과 동일하게(resb=0.5/res015=0.15/res03=0.3)")
ap.add_argument("--max_steps", type=int, default=300)
ap.add_argument("--hold", type=float, default=3.0)
ap.add_argument("--goal_as_target", action="store_true", help="켜면 목표=골대 방향(기본은 학습 분포=무작위 방향)")
a = ap.parse_args()

with open(a.vecnorm, "rb") as f:
    vn = pickle.load(f)
obs_rms, eps, clip = vn.obs_rms, vn.epsilon, vn.clip_obs


def norm_obs(o):
    o = np.asarray(o, dtype=np.float32)
    return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + eps), -clip, clip).astype(np.float32)


model = PPO.load(a.model, device="cpu")
env = OP3KickEnvV6Res(kick_window=None, max_steps=a.max_steps,
                      goal_as_target=a.goal_as_target, goal_randomize=False,
                      res_scale=a.res_scale)
mj_model, mj_data = env.model, env.data
print("model :", a.model, " res_scale=", a.res_scale,
      " — 자동 반복, Ctrl+C로 종료. obs_dim:", env.observation_space.shape)
with mujoco.viewer.launch_passive(mj_model, mj_data) as v:
    while v.is_running():
        o, info = env.reset(); env.reset_bonus()
        print("theta=%.2f (명령 방향, rad)" % info["theta"])
        for _ in range(a.max_steps):
            act, _ = model.predict(norm_obs(o), deterministic=True)
            o, _, te, tr, inf = env.step(act)
            v.sync(); time.sleep(0.03)
            if not v.is_running() or te or tr:
                break
        t0 = time.time()
        while time.time() - t0 < a.hold and v.is_running():
            v.sync(); time.sleep(0.03)
