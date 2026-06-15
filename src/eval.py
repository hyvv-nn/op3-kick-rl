# eval.py  v2  [헤드라인 — in-distribution 교차점 곡선]
# 해석적 차기 vs RL을 U0~U4에서 N회 평가 → 교차점 곡선(CSV+PNG). 직진 차기(goal_cond=False).
# 실행: python eval.py --model runs/op3_kick_ppo_s0.zip --vecnorm runs/vecnorm_s0.pkl --N 20 --params "{'amp_hip':1.0,'amp_knee':0.2,'t_swing':0.15}"
# ★ 지표 메모: 마찰 무작위화로 '직진 비거리'는 불확실성↑에 오히려 커질 수 있다(낮은 마찰=더 구름).
#   교차점 서사가 더 깨끗한 지표 = 목표존 성공률 또는 aim_err(목표-상대). proj와 함께 같이 보라.
import argparse
import ast
import csv
import numpy as np
from op3_kick_env import OP3KickEnv
from domain_randomization import DomainRandomize, LEVELS
from analytic_kick import make_kick


def ev_analytic(level, N, params, max_steps=160, dt=0.025):
    env = DomainRandomize(OP3KickEnv(goal_cond=False, max_steps=max_steps), level=level)
    ds, errs = [], []
    for s in range(N):
        env.reset(seed=1000 + s); env.unwrapped.reset_bonus()
        k = make_kick(**params); t = 0.0; info = {"proj": 0.0, "aim_err_deg": 180.0}
        for _ in range(max_steps):
            _, _, te, tr, info = env.step(k(t)); t += dt
            if te or tr:
                break
        ds.append(info["proj"]); errs.append(info["aim_err_deg"])
    return float(np.mean(ds)), float(np.std(ds)), float(np.mean(errs))


def _norm(o, vn):
    return o if vn is None else np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                                        -vn.clip_obs, vn.clip_obs).astype(np.float32)


def ev_rl(level, N, model, vn, max_steps=160):
    env = DomainRandomize(OP3KickEnv(goal_cond=False, max_steps=max_steps), level=level)
    ds, errs = [], []
    for s in range(N):
        o, _ = env.reset(seed=2000 + s); env.unwrapped.reset_bonus(); info = {"proj": 0.0, "aim_err_deg": 180.0}
        for _ in range(max_steps):
            a, _ = model.predict(_norm(o, vn), deterministic=True)
            o, _, te, tr, info = env.step(a)
            if te or tr:
                break
        ds.append(info["proj"]); errs.append(info["aim_err_deg"])
    return float(np.mean(ds)), float(np.std(ds)), float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--vecnorm")
    ap.add_argument("--N", type=int, default=20); ap.add_argument("--params", default="{}")
    args = ap.parse_args()
    params = ast.literal_eval(args.params) if args.params else {}
    model = vn = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        if args.vecnorm:
            import pickle
            vn = pickle.load(open(args.vecnorm, "rb"))

    levels = sorted([L for L in LEVELS if L <= 4])
    rows = [("U", "analytic_mean", "analytic_std", "rl_mean", "rl_std")]
    for L in levels:
        am, asd, _ = ev_analytic(L, args.N, params)
        if model is not None:
            rm, rsd, _ = ev_rl(L, args.N, model, vn)
        else:
            rm, rsd = float("nan"), float("nan")
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
        plt.errorbar(U, am, yerr=asd, marker="o", label="Analytic kick (classical)")
        if model is not None:
            plt.errorbar(U, rm, yerr=rsd, marker="s", label="PPO+DR (learned)")
        plt.xlabel("Environment uncertainty level U")
        plt.ylabel("Ball travel toward target (m)")
        plt.title("OP3 fixed-base kick: classical vs learned (crossover)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("crossover.png", dpi=150)
        print("saved crossover.csv, crossover.png")
    except Exception as e:
        print("plot skipped:", e, "(crossover.csv saved)")


if __name__ == "__main__":
    main()
