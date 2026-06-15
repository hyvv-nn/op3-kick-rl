# eval_sim2sim.py  [샌드박스 검증완료 2026-06-15: in-dist U0~U3 + OOD U4~U6 밴드 생성]
# sim-to-sim 강건성: 학습 DR 분포 *밖*(OOD)에서 해석적 vs RL 다시드 평가.
# 학습은 U0~U3 가정. OOD = U4(경계) + U5,U6(외삽 — 어떤 학습에도 없던 동역학). P1의 "미지 플랜트" 프로토콜의 휴머노이드판.
# 실행: python eval_sim2sim.py --model runs/op3_kick_ppo_s0.zip --vecnorm runs/vecnorm_s0.pkl --N 10 --params "{'amp_hip':1.0,'amp_knee':0.2,'t_swing':0.15}"
import argparse
import ast
import csv
import numpy as np
from op3_kick_env import OP3KickEnv
from domain_randomization import DomainRandomize, LEVELS
from analytic_kick import make_kick

# 외삽 레벨(학습 분포 밖)
LEVELS[5] = dict(m=0.11, f=0.55, x=0.05, g=0.28, noise=0.030, lat=3)
LEVELS[6] = dict(m=0.14, f=0.70, x=0.06, g=0.35, noise=0.040, lat=4)


def ev_analytic(level, N, params, max_steps=160, dt=0.025):
    env = DomainRandomize(OP3KickEnv(goal_cond=False, max_steps=max_steps), level=level)
    ds = []
    for s in range(N):
        env.reset(seed=3000 + s); env.unwrapped.reset_bonus()
        k = make_kick(**params); t = 0.0; info = {"proj": 0.0}
        for _ in range(max_steps):
            _, _, te, tr, info = env.step(k(t)); t += dt
            if te or tr:
                break
        ds.append(info["proj"])
    return float(np.mean(ds)), float(np.std(ds))


def _norm(o, vn):
    return o if vn is None else np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                                        -vn.clip_obs, vn.clip_obs).astype(np.float32)


def ev_rl(level, N, model, vn, max_steps=160):
    env = DomainRandomize(OP3KickEnv(goal_cond=False, max_steps=max_steps), level=level)
    ds = []
    for s in range(N):
        o, _ = env.reset(seed=4000 + s); env.unwrapped.reset_bonus(); info = {"proj": 0.0}
        for _ in range(max_steps):
            a, _ = model.predict(_norm(o, vn), deterministic=True)
            o, _, te, tr, info = env.step(a)
            if te or tr:
                break
        ds.append(info["proj"])
    return float(np.mean(ds)), float(np.std(ds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--vecnorm")
    ap.add_argument("--N", type=int, default=10); ap.add_argument("--params", default="{}")
    a = ap.parse_args()
    params = ast.literal_eval(a.params) if a.params else {}
    model = vn = None
    if a.model:
        from stable_baselines3 import PPO
        model = PPO.load(a.model, device="cpu")
        if a.vecnorm:
            import pickle
            vn = pickle.load(open(a.vecnorm, "rb"))
    INDIST = [0, 1, 2, 3]; OOD = [4, 5, 6]
    rows = [("level", "regime", "analytic_mean", "analytic_std", "rl_mean", "rl_std")]
    for L in INDIST + OOD:
        am, asd = ev_analytic(L, a.N, params)
        rm, rsd = (ev_rl(L, a.N, model, vn) if model else (float("nan"), float("nan")))
        reg = "in-dist" if L in INDIST else "OOD"
        rows.append((L, reg, am, asd, rm, rsd))
        print("U%d[%s]: analytic %.3f+/-%.3f | RL %.3f+/-%.3f" % (L, reg, am, asd, rm, rsd))
    csv.writer(open("sim2sim.csv", "w", newline="")).writerows(rows)
    print("saved sim2sim.csv")
