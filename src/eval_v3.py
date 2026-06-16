# eval_v3.py  [미검증 스켈레톤 — 다음 세션이 실 모델로 검증/캘리브레이션]
# v3 평가축: raw 비거리 → 방향정확도(aim_err)·성공률·신뢰성(분산). 공 정지(coast)까지 본 뒤 최종위치로 계산.
# 사용: v2 env/DR/analytic 옆(robotis_op3\)에 두고 실행.
#   python eval_v3.py --model runs\op3_kick_s0_g0_4999200_steps.zip --vecnorm runs\vecnorm_s0_g0.pkl \
#          --N 20 --params "{'amp_hip':1.0,'amp_knee':0.2,'t_swing':0.15}"
import argparse, ast, csv
import numpy as np
from op3_kick_env import OP3KickEnv
from domain_randomization import DomainRandomize, LEVELS
from analytic_kick import make_kick

# ★ 캘리브레이션 대상: analytic이 U0에서 성공률 ~80~100%, U4에서 급락하도록 1회 조정
ANG_TOL = 25.0      # deg : 방향 허용 오차
D_MIN   = 0.6       # m   : 목표방향 최소 도달거리(골라인)
MAXSTEPS = 350      # 공 정지까지 coast 포함(에피소드 길게)
DT = 0.025


def _final(u):
    disp = u.data.qpos[u.bq:u.bq + 2].copy() - u._ball0
    cmd = u.cmd_dir
    nd = float(np.linalg.norm(disp))
    proj = float(np.dot(disp, cmd))
    ang = float(np.degrees(np.arccos(np.clip(np.dot(disp / (nd + 1e-8), cmd), -1, 1)))) if nd > 0.05 else 180.0
    return proj, ang


def run_analytic(level, N, params, goal_cond):
    env = DomainRandomize(OP3KickEnv(goal_cond=goal_cond, max_steps=MAXSTEPS), level=level)
    u = env.unwrapped; P, A = [], []
    for s in range(N):
        _, info = env.reset(seed=1000 + s); u.reset_bonus()
        k = make_kick(aim=info.get("cmd_theta", 0.0), **params); t = 0.0
        for _ in range(MAXSTEPS):
            _, _, te, tr, _ = env.step(k(t)); t += DT
            if te or tr:
                break
        proj, ang = _final(u); P.append(proj); A.append(ang)
    return np.array(P), np.array(A)


def _norm(o, vn):
    return o if vn is None else np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                                        -vn.clip_obs, vn.clip_obs).astype(np.float32)


def run_rl(level, N, model, vn, goal_cond):
    env = DomainRandomize(OP3KickEnv(goal_cond=goal_cond, max_steps=MAXSTEPS), level=level)
    u = env.unwrapped; P, A = [], []
    for s in range(N):
        o, _ = env.reset(seed=2000 + s); u.reset_bonus()
        for _ in range(MAXSTEPS):
            a, _ = model.predict(_norm(o, vn), deterministic=True)
            o, _, te, tr, _ = env.step(a)
            if te or tr:
                break
        proj, ang = _final(u); P.append(proj); A.append(ang)
    return np.array(P), np.array(A)


def summarize(P, A):
    succ = float(np.mean((A <= ANG_TOL) & (P >= D_MIN)))
    return dict(success=succ, aim_mean=float(np.mean(A)), aim_std=float(np.std(A)),
                proj_mean=float(np.mean(P)), proj_std=float(np.std(P)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--vecnorm")
    ap.add_argument("--N", type=int, default=20); ap.add_argument("--params", default="{}")
    ap.add_argument("--goal_cond", type=int, default=0)
    a = ap.parse_args()
    params = ast.literal_eval(a.params) if a.params else {}
    gc = bool(a.goal_cond)
    model = vn = None
    if a.model:
        from stable_baselines3 import PPO
        model = PPO.load(a.model, device="cpu")
        if a.vecnorm:
            import pickle
            vn = pickle.load(open(a.vecnorm, "rb"))

    levels = sorted([L for L in LEVELS if L <= 4])
    rows = [("U", "a_success", "a_aim", "a_aimstd", "a_projstd", "r_success", "r_aim", "r_aimstd", "r_projstd")]
    AS, RS = {}, {}
    for L in levels:
        aP, aA = run_analytic(L, a.N, params, gc); asum = summarize(aP, aA)
        if model is not None:
            rP, rA = run_rl(L, a.N, model, vn, gc); rsum = summarize(rP, rA)
        else:
            rsum = dict(success=float("nan"), aim_mean=float("nan"), aim_std=float("nan"), proj_std=float("nan"))
        AS[L], RS[L] = asum, rsum
        rows.append((L, asum["success"], asum["aim_mean"], asum["aim_std"], asum["proj_std"],
                     rsum["success"], rsum["aim_mean"], rsum["aim_std"], rsum["proj_std"]))
        print("U%d | analytic succ=%.2f aim=%.1f±%.1f projσ=%.2f | RL succ=%.2f aim=%.1f±%.1f projσ=%.2f"
              % (L, asum["success"], asum["aim_mean"], asum["aim_std"], asum["proj_std"],
                 rsum["success"], rsum["aim_mean"], rsum["aim_std"], rsum["proj_std"]))
    with open("crossover_v3.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        U = levels
        def plot(yfun, ylabel, fname, err=None):
            plt.figure(figsize=(6, 4))
            ya = [yfun(AS[L]) for L in U]
            plt.plot(U, ya, "o-", label="Analytic (classical)")
            if model is not None:
                yr = [yfun(RS[L]) for L in U]; plt.plot(U, yr, "s-", label="PPO+DR (learned)")
            plt.xlabel("Environment uncertainty level U"); plt.ylabel(ylabel)
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
        plot(lambda d: d["success"], "Success rate (dir+reach)", "success_vs_U.png")
        plot(lambda d: d["aim_mean"], "Aim error (deg)", "aim_err_vs_U.png")
        plot(lambda d: d["proj_std"], "Std of distance (reliability)", "reliability_vs_U.png")
        print("saved crossover_v3.csv, success_vs_U.png, aim_err_vs_U.png, reliability_vs_U.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
