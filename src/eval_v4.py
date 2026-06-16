# eval_v4.py  [미검증 스켈레톤 — 다음 세션이 재학습 후 검증·캘리브레이션]
# 명령 일반화 경계: 명령(방향) 범위를 넓혀가며 「fixed analytic(단일설계) vs oracle analytic(목표별 재설계) vs RL(목표조건부)」 비교.
# 예상: 범위↑ → fixed 성공률 붕괴, oracle 높음, RL ≈ oracle 유지 → "하나의 학습 정책 = 목표별 재설계 고전에 필적, 단일설계 압도".
# v4 env/analytic 옆(robotis_op3\)에 두고 실행:
#   python eval_v4.py --model runs\op3_kick_v4_s0.zip --vecnorm runs\vecnorm_v4_s0.pkl --N 20
import argparse, csv
import numpy as np
from env_v4 import OP3KickEnvV4
from analytic_kick import make_kick     # v2 analytic (amp_hip, aim 인자)

SUCC_R = 0.4        # 목표 반경(m) — 캘리브레이션 대상
MAXSTEPS = 350      # 공 정지까지 coast
DT = 0.025
BASE = dict(amp_knee=0.2, t_swing=0.15)
D_FIX = 1.4         # 방향 일반화 실험에선 거리 고정
D_REF = 1.4


def power_for_D(D):  # 오라클 파워-거리 매핑 [★캘리브레이션 필요: D별 amp→실제 도달거리]
    return float(np.clip(D / D_REF, 0.5, 2.0))


def _norm(o, vn):
    return o if vn is None else np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                                        -vn.clip_obs, vn.clip_obs).astype(np.float32)


def run(mode, N, aim_range, model=None, vn=None):
    env = OP3KickEnvV4(kick_window=None, max_steps=MAXSTEPS, aim_range=aim_range, d_min=D_FIX, d_max=D_FIX)
    errs, succ = [], 0
    for s in range(N):
        o, info = env.reset(seed=4000 + s); env.reset_bonus()
        th, D = info["theta"], info["D"]; inf = {"final_err": 9.9}
        if mode == "rl":
            for _ in range(MAXSTEPS):
                a, _ = model.predict(_norm(o, vn), deterministic=True)
                o, _, te, tr, inf = env.step(a)
                if te or tr:
                    break
        else:
            k = make_kick(amp_hip=1.0, aim=0.0, **BASE) if mode == "fixed" \
                else make_kick(amp_hip=power_for_D(D), aim=th, **BASE)   # oracle: 목표 방향·파워 사용
            t = 0.0
            for _ in range(MAXSTEPS):
                _, _, te, tr, inf = env.step(k(t)); t += DT
                if te or tr:
                    break
        errs.append(inf["final_err"]); succ += int(inf["final_err"] < SUCC_R)
    return succ / N, float(np.mean(errs)), float(np.std(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--vecnorm"); ap.add_argument("--N", type=int, default=20)
    a = ap.parse_args()
    model = vn = None
    if a.model:
        from stable_baselines3 import PPO
        model = PPO.load(a.model, device="cpu")
        if a.vecnorm:
            import pickle
            vn = pickle.load(open(a.vecnorm, "rb"))

    ranges = [0.0, 0.15, 0.30, 0.45, 0.60]   # 명령(방향) 범위 폭 Θ
    rows = [("aim_range", "fixed_succ", "oracle_succ", "rl_succ", "fixed_err", "oracle_err", "rl_err")]
    data = {"fixed": [], "oracle": [], "rl": []}
    for R in ranges:
        fs, fe, _ = run("fixed", a.N, R)
        os_, oe, _ = run("oracle", a.N, R)
        rs, re, _ = (run("rl", a.N, R, model, vn) if model else (float("nan"), float("nan"), 0.0))
        data["fixed"].append(fs); data["oracle"].append(os_); data["rl"].append(rs)
        rows.append((R, fs, os_, rs, fe, oe, re))
        print("Θ=%.2f | fixed succ=%.2f err=%.2f | oracle succ=%.2f err=%.2f | RL succ=%.2f err=%.2f"
              % (R, fs, fe, os_, oe, rs, re))
    with open("crossover_v4.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(ranges, data["fixed"], "o-", label="Fixed analytic (single design)")
        plt.plot(ranges, data["oracle"], "^-", label="Oracle analytic (per-goal redesign)")
        if model is not None:
            plt.plot(ranges, data["rl"], "s-", label="Goal-conditioned RL (one policy)")
        plt.xlabel("Commanded direction range Θ (rad)"); plt.ylabel("Success rate (reach target)")
        plt.title("Command generalization: one policy vs single/per-goal classical")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("success_vs_cmdrange.png", dpi=150)
        print("saved crossover_v4.csv, success_vs_cmdrange.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
