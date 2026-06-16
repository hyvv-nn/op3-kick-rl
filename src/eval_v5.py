# eval_v5.py  [미검증 스켈레톤 — v5 정책 확보(단발킥 게이트 통과) 후 실행하는 *다운스트림* 명령 일반화 평가]
# 선행조건: v5 학습 → watch_kick으로 "단발 조준킥" 확인(상체 안 휘적·공 안 깔림). 그 전엔 이 평가 무의미.
# 내용: (방향 θ × 거리 D) 2D 격자에서 「fixed analytic(단일설계) vs oracle analytic(셀별 재설계) vs goal-cond RL」 성공률 heatmap.
#   - RL    : env_v5(full_action=False) — 마스킹된 6-dim 정책
#   - analytic: env_v5(full_action=True) — 20-dim 해석적 킥을 같은 씬에서 구동(공정 비교)
# 예상: fixed=한 점(섬), oracle·RL=면(≈). v4(1D, eval_v4.py)의 2D 확장.
#   python eval_v5.py --model runs\op3_kick_v5_s0.zip --vecnorm runs\vecnorm_v5_s0.pkl --N 12
import argparse, csv
import numpy as np
from env_v5b import OP3KickEnvV5B       # v5-B: 물리적 고정 + 지지자세 (v5-A 자빠짐 대응)
from analytic_kick import make_kick     # v2 analytic (amp_hip, aim 인자; 20-dim action)

SUCC_R = 0.4        # 목표 반경(m) — 캘리브레이션 대상(v4와 공유)
MAXSTEPS = 350      # 공 정지까지 coast
DT = 0.025
D_REF = 1.4
BASE = dict(amp_knee=0.2, t_swing=0.15)

THETAS = [-0.4, -0.2, 0.0, 0.2, 0.4]      # rad
DISTS  = [1.0, 1.2, 1.4, 1.6, 1.8]        # m


def power_for_D(D):  # 오라클 파워-거리 매핑 [★캘리브레이션 필요]
    return float(np.clip(D / D_REF, 0.5, 2.0))


def _norm(o, vn):
    return o if vn is None else np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                                        -vn.clip_obs, vn.clip_obs).astype(np.float32)


def _pin_target(env, th, D):
    """env.reset 후 (θ,D)를 결정론적으로 고정하고, 보정된 obs 반환."""
    env.theta, env.D = th, D
    env.target = env.ball0 + D * np.array([np.cos(th), np.sin(th)])
    env._prevdist = float(np.linalg.norm(env.target - env.ball0))
    return env._obs()


def run_cell(mode, th, D, N, model=None, vn=None, cell_seed=0):
    full = (mode != "rl")                       # 해석적=full_action(20), RL=마스킹(6)
    env = OP3KickEnvV5B(kick_window=None, max_steps=MAXSTEPS,
                        aim_range=abs(th) + 1e-3, d_min=D, d_max=D, full_action=full)
    errs, succ = [], 0
    for s in range(N):
        env.reset(seed=cell_seed + s); env.reset_bonus()
        o = _pin_target(env, th, D)
        inf = {"final_err": 9.9}
        if mode == "rl":
            for _ in range(MAXSTEPS):
                a, _ = model.predict(_norm(o, vn), deterministic=True)
                o, _, te, tr, inf = env.step(a)
                if te or tr:
                    break
        else:
            k = make_kick(amp_hip=1.0, aim=0.0, **BASE) if mode == "fixed" \
                else make_kick(amp_hip=power_for_D(D), aim=th, **BASE)
            t = 0.0
            for _ in range(MAXSTEPS):
                _, _, te, tr, inf = env.step(k(t)); t += DT
                if te or tr:
                    break
        errs.append(inf["final_err"]); succ += int(inf["final_err"] < SUCC_R)
    return succ / N, float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--vecnorm"); ap.add_argument("--N", type=int, default=12)
    a = ap.parse_args()
    model = vn = None
    if a.model:
        from stable_baselines3 import PPO
        model = PPO.load(a.model, device="cpu")
        if a.vecnorm:
            import pickle
            vn = pickle.load(open(a.vecnorm, "rb"))

    nT, nD = len(THETAS), len(DISTS)
    modes = ["fixed", "oracle"] + (["rl"] if model else [])
    S = {m: np.full((nD, nT), np.nan) for m in modes}
    E = {m: np.full((nD, nT), np.nan) for m in modes}
    rows = [("theta", "D", "fixed_succ", "oracle_succ", "rl_succ", "fixed_err", "oracle_err", "rl_err")]

    for j, D in enumerate(DISTS):
        for i, th in enumerate(THETAS):
            cseed = 5000 + (j * nT + i) * 100
            cell = {}
            for m in modes:
                sc, er = run_cell(m, th, D, a.N, model, vn, cseed)
                S[m][j, i] = sc; E[m][j, i] = er; cell[m] = (sc, er)
            rows.append((th, D,
                         cell.get("fixed", (float("nan"),))[0], cell.get("oracle", (float("nan"),))[0],
                         cell.get("rl", (float("nan"),))[0],
                         cell.get("fixed", (0, float("nan")))[1], cell.get("oracle", (0, float("nan")))[1],
                         cell.get("rl", (0, float("nan")))[1]))
            print("θ=%+.2f D=%.1f | " % (th, D)
                  + " | ".join("%s succ=%.2f err=%.2f" % (m, cell[m][0], cell[m][1]) for m in modes))

    with open("crossover_v5_2d.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    for m in modes:
        print("[mean succ] %-7s = %.3f" % (m, np.nanmean(S[m])))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def heatmap(metric, fname, label, vmin, vmax, cmap):
            fig, axes = plt.subplots(1, len(modes), figsize=(4.2 * len(modes), 3.6), squeeze=False)
            for ax, m in zip(axes[0], modes):
                im = ax.imshow(metric[m], origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap,
                               extent=[min(THETAS), max(THETAS), min(DISTS), max(DISTS)])
                ax.set_title({"fixed": "Fixed analytic", "oracle": "Oracle analytic",
                              "rl": "Goal-cond RL"}[m])
                ax.set_xlabel("commanded theta (rad)"); ax.set_ylabel("commanded D (m)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(label); fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)

        heatmap(S, "success_heatmap_2d.png", "2D command generalization - success rate", 0, 1, "viridis")
        heatmap(E, "error_heatmap_2d.png", "2D command generalization - target error (m)", 0, None, "magma_r")
        print("saved crossover_v5_2d.csv, success_heatmap_2d.png, error_heatmap_2d.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
