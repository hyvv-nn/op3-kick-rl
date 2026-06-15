# analytic_tune.py  [샌드박스 검증완료 2026-06-15: BEST U0 proj=0.852, amp_hip=1.0,amp_knee=0.2,t_swing=0.15]
# U0에서 비거리(목표방향 proj) 최대가 되는 해석적 차기 파라미터 그리드 탐색 → 강한 고전 baseline 확보.
# (약한 대조군을 이기는 건 의미 없다. 이 튜닝이 비교의 공정성을 만든다.)
import itertools
import numpy as np
from op3_kick_env import OP3KickEnv
from analytic_kick import make_kick


def score(params, N=3, max_steps=160, dt=0.025):
    env = OP3KickEnv(goal_cond=False, max_steps=max_steps)
    ds = []
    for s in range(N):
        env.reset(seed=s); env.reset_bonus()
        k = make_kick(**params); t = 0.0; info = {"proj": 0.0}
        for _ in range(max_steps):
            _, _, te, tr, info = env.step(k(t)); t += dt
            if te or tr:
                break
        ds.append(info["proj"])
    return float(np.mean(ds))


def tune():
    best = None
    grid = dict(amp_hip=[1.0, 1.3, 1.6], amp_knee=[0.2, 0.4, 0.6], t_swing=[0.15, 0.20])
    for ah, ak, ts in itertools.product(*grid.values()):
        p = dict(amp_hip=ah, amp_knee=ak, t_swing=ts)
        s = score(p)
        if best is None or s > best[0]:
            best = (s, p)
    return best


if __name__ == "__main__":
    s, p = tune()
    print("BEST U0 proj=%.3f params=%s" % (s, p))
    print("→ 이 params를 eval.py/eval_sim2sim.py의 --params 로, analytic_kick.make_kick 기본값으로 사용")
