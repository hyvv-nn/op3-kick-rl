# watch_analytic.py — 고전(해석적 임펄스-최대화) 킥 시각화. 정책 없음(make_kick만 구동).
#   학습 킥(watch_v6 / watch_v6res)과의 *대비*용: 고전은 "정밀하지만 약함"을 눈으로 본다.
#   기본 oracle = 방향마다 aim 재설계(정밀). --fixed = 단일설계(조준 안 함, 헛발질 보기).
# 실행 예 (venv python으로):
#   & $py watch_analytic.py --hold 4
#   & $py watch_analytic.py --fixed --hold 4     # 단일설계(조준 못 함) 대비
import argparse, time
import numpy as np, mujoco, mujoco.viewer
from env_v6 import OP3KickEnvV6
from analytic_kick import make_kick

ap = argparse.ArgumentParser()
ap.add_argument("--max_steps", type=int, default=200)
ap.add_argument("--hold", type=float, default=3.0)
ap.add_argument("--fixed", action="store_true", help="단일설계(aim=0 고정). 기본은 oracle(방향별 재설계)")
ap.add_argument("--amp_hip", type=float, default=1.0, help="킥 강도(해석적 baseline). 실측 튜닝값 1.0")
a = ap.parse_args()

DT = 0.025                         # frame_skip(5) * timestep(0.005) = env 1스텝당 실시간(초)
BASE = dict(amp_knee=0.2, t_swing=0.15)

env = OP3KickEnvV6(kick_window=None, max_steps=a.max_steps, full_action=True)
mj_model, mj_data = env.model, env.data
mode = "fixed(단일설계·조준X)" if a.fixed else "oracle(방향별 재설계·정밀)"
print("ANALYTIC 고전 킥 :", mode, " — 정책 없음. 자동 반복, Ctrl+C로 종료. obs_dim:", env.observation_space.shape)
with mujoco.viewer.launch_passive(mj_model, mj_data) as v:
    while v.is_running():
        o, info = env.reset(); env.reset_bonus()
        th = float(info["theta"])                       # 이번 에피소드 명령 방향(rad)
        k = make_kick(amp_hip=a.amp_hip, aim=(0.0 if a.fixed else th), **BASE)
        print("theta=%.2f (명령 방향)  mode=%s" % (th, "fixed" if a.fixed else "oracle"))
        t = 0.0
        for _ in range(a.max_steps):
            o, _, te, tr, inf = env.step(k(t)); t += DT
            v.sync(); time.sleep(0.03)
            if not v.is_running() or te or tr:
                break
        t0 = time.time()
        while time.time() - t0 < a.hold and v.is_running():
            v.sync(); time.sleep(0.03)
