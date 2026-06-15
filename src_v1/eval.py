# eval.py  v1 [초기 버전 — 직진 비거리(ball_x) 교차점. 정식=상위 폴더 v2(성공률·정확도 지표)]
# 해석적 vs RL을 U0~U4에서 N회 평가 → 교차점 곡선(CSV+PNG).
# 실행: python eval.py --model runs/op3_kick_ppo_s0.zip --vecnorm runs/vecnorm_s0.pkl --N 20
import argparse
import csv
import numpy as np
from op3_kick_env import OP3KickEnv
from domain_randomization import DomainRandomize, LEVELS
from analytic_kick import make_kick


def eval_analytic(level, N, kick, max_steps=200, dt=0.025):
    env = DomainRandomize(OP3KickEnv(max_steps=max_steps), level=level)
    ds = []
    for s in range(N):
        env.reset(seed=1000 + s); t = 0.0; info = {"ball_x": 0.0}
        for _ in range(max_steps):
            _, _, te, tr, info = env.step(kick(t)); t += dt
            if te or tr:
                break
        ds.append(info["ball_x"])
    return float(np.mean(ds)), float(np.std(ds))


def _norm_obs(o, vn):
    if vn is None:
        return o
    return np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                   -vn.clip_obs, vn.clip_obs).astype(np.float32)


def eval_rl(level, N, model, vn=None, max_steps=200):
    env = DomainRandomize(OP3KickEnv(max_steps=max_steps), level=level)
    ds = []
    for s in range(N):
        o, _ = env.reset(seed=2000 + s); info = {"ball_x": 0.0}
        for _ in range(max_steps):
            a, _ = model.predict(_norm_obs(o, vn), deterministic=True)
            o, _, te, tr, info = env.step(a)
            if te or tr:
                break
        ds.append(info["ball_x"])
    return float(np.mean(ds)), float(np.std(ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--vecnorm"); ap.add_argument("--N", type=int, default=20)
    args = ap.parse_args()
    kick = make_kick()
    model = vn = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        if args.vecnorm:
            import pickle
            with open(args.vecnorm, "rb") as f:
                vn = pickle.load(f)
    levels = sorted(LEVELS.keys())
    rows = [("U", "analytic_mean", "analytic_std", "rl_mean", "rl_std")]
    for L in levels:
        am, asd = eval_analytic(L, args.N, kick)
        rm, rsd = (eval_rl(L, args.N, model, vn) if model else (float("nan"), float("nan")))
        rows.append((L, am, asd, rm, rsd))
        print("U%d: analytic %.3f+/-%.3f | RL %.3f+/-%.3f" % (L, am, asd, rm, rsd))
    with open("crossover.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        U = levels
        am = [rows[i + 1][1] for i in range(len(U))]; asd = [rows[i + 1][2] for i in range(len(U))]
        rm = [rows[i + 1][3] for i in range(len(U))]; rsd = [rows[i + 1][4] for i in range(len(U))]
        plt.figure(figsize=(6, 4))
        plt.errorbar(U, am, yerr=asd, marker="o", label="Analytic kick")
        if model is not None:
            plt.errorbar(U, rm, yerr=rsd, marker="s", label="PPO+DR")
        plt.xlabel("Uncertainty level U"); plt.ylabel("Ball forward distance (m)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("crossover.png", dpi=150)
        print("saved crossover.csv, crossover.png")
    except Exception as e:
        print("plot skipped:", e, "(crossover.csv saved)")


if __name__ == "__main__":
    main()
