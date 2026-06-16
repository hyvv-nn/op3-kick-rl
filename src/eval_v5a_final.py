# eval_v5a_final.py  [위치(qpos) 기반 — 속도 미사용, 인덱싱 함정 회피]
# ─────────────────────────────────────────────────────────────────────────────
# 측정 대장정 결론(2026-06-16, 7회 진단):
#   · v5-A 실측: 킥파워 ~7m(analytic 1.86m의 ~4배), 방향 조준 O(거침: +8° 바이어스·gain~60%),
#     거리(D) 조절 X(둘 다 고정파워), lateral은 θ 따라 이동.
#   · 속도(qvel) 기반 측정은 free-joint dof 인덱싱 함정으로 전부 오염 → 위치(qpos)만 사용.
#
# 헤드라인 서사(정직): "학습=강력하나 거친 조준 / 고전=약하나 정밀. 강점이 상보적 = P1 crossover의 휴머노이드 확장."
#
# 3지표(전부 위치 변위 disp = qpos - ball0 에서 유도):
#   ① aim_err(deg)   = |실제방향 − 명령 θ|           ← 조준 정확도 (analytic 정밀 vs RL 거침)
#   ② reach(m)       = proj (명령방향 투영거리)        ← 킥 파워 (RL 압도)
#   ③ goal_dir(0/1)  = (proj ≥ MIN_REACH) and (aim_err < ANG_TOL)  ← 방향 골 성공률(거리무관 각도기준, 멀리 차도 공정)
#
# 비교군: fixed(단일설계 analytic) / oracle(목표별 재설계 analytic) / rl(v5-A 마스킹)
#   analytic = full_action=True(20관절), RL = full_action=False(6관절 마스킹). 같은 씬, 공정.
#
# 실행:
#   python eval_v5a_final.py --model runs\op3_kick_v5_s0.zip --vecnorm runs\vecnorm_v5_s0.pkl --N 8
#   (v5.5 기각근거) ... --model runs\op3_kick_v55_s0.zip --vecnorm runs\vecnorm_v55_s0.pkl --tag v55
# ─────────────────────────────────────────────────────────────────────────────
import argparse, csv
import numpy as np
from env_v5 import OP3KickEnvV5
from analytic_kick import make_kick

THETAS = [-0.4, -0.2, 0.0, 0.2, 0.4]      # rad — 방향 명령 격자(학습분포 ±0.4)
D_FIX   = 1.4        # 거리 명령 고정(D 조절은 둘 다 불가로 확인됨 → θ축에 집중)
MAXSTEPS = 200      # 위치 기반이라 coast 폭주 무관, 200이면 충분히 정착
DT = 0.025          # env frame_skip(5)*timestep(0.005) 정합 확인됨
D_REF = 1.4
BASE = dict(amp_knee=0.2, t_swing=0.15)

MIN_REACH_DEFAULT = 0.6     # m   : 골 인정 최소 전진(헛발질 배제). 캘리브 대상.
ANG_TOL_DEFAULT   = 15.0    # deg : 방향 골 허용오차(거리무관 — 멀리 차도 불이익 없음). 캘리브 대상.


def power_for_D(D):
    return float(np.clip(D / D_REF, 0.5, 2.0))


def _norm(o, vn):
    if vn is None:
        return o
    return np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                   -vn.clip_obs, vn.clip_obs).astype(np.float32)


def _metrics(u, th):
    """위치 변위 기반: (proj, aim_err_deg, lateral). 속도 미사용."""
    disp = u.data.qpos[u.bq:u.bq + 2].copy() - u.ball0
    cmd = np.array([np.cos(th), np.sin(th)])
    nd = float(np.linalg.norm(disp))
    proj = float(np.dot(disp, cmd))
    lateral = float(np.linalg.norm(disp - proj * cmd))
    if nd > 0.05:
        actual_ang = np.degrees(np.arctan2(disp[1], disp[0]))
        aim_err = abs(actual_ang - np.degrees(th))
    else:
        aim_err = 180.0
    return proj, aim_err, lateral


def _set_target(u, th, D):
    u.theta, u.D = th, D
    u.target = u.ball0 + D * np.array([np.cos(th), np.sin(th)])
    u._prevdist = float(np.linalg.norm(u.target - u.ball0))


def run_cell(mode, th, N, model, vn, cell_seed, min_reach, ang_tol):
    full = (mode != "rl")
    env = OP3KickEnvV5(kick_window=None, max_steps=MAXSTEPS,
                       aim_range=abs(th) + 1e-3, d_min=D_FIX, d_max=D_FIX, full_action=full)
    u = env.unwrapped
    PR, AE, LA = [], [], []
    for s in range(N):
        env.reset(seed=cell_seed + s); env.reset_bonus()
        _set_target(u, th, D_FIX)
        if mode == "rl":
            for _ in range(MAXSTEPS):
                o = u._obs()
                a, _ = model.predict(_norm(o, vn), deterministic=True)
                env.step(a)
        else:
            k = make_kick(amp_hip=1.0, aim=0.0, **BASE) if mode == "fixed" \
                else make_kick(amp_hip=power_for_D(D_FIX), aim=th, **BASE)
            t = 0.0
            for _ in range(MAXSTEPS):
                env.step(k(t)); t += DT
        proj, aim_err, lat = _metrics(u, th)
        PR.append(proj); AE.append(aim_err); LA.append(lat)
    PR, AE, LA = np.array(PR), np.array(AE), np.array(LA)
    # ★ 골 판정 = 거리무관 각도 기준(멀리 차는 정책 불이익 제거): 전진 + 방향오차 한도
    goal = (PR >= min_reach) & (AE < ang_tol)
    return dict(reach_mean=float(PR.mean()), reach_std=float(PR.std()),
                aim_mean=float(AE.mean()), aim_std=float(AE.std()),
                lat_mean=float(LA.mean()),
                goal_rate=float(goal.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--vecnorm")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--min_reach", type=float, default=MIN_REACH_DEFAULT)
    ap.add_argument("--ang_tol", type=float, default=ANG_TOL_DEFAULT)
    ap.add_argument("--tag", type=str, default="v5a")
    a = ap.parse_args()

    model = vn = None
    if a.model:
        from stable_baselines3 import PPO
        model = PPO.load(a.model, device="cpu")
        if a.vecnorm:
            import pickle
            vn = pickle.load(open(a.vecnorm, "rb"))

    modes = ["fixed", "oracle"] + (["rl"] if model else [])
    R = {m: [] for m in modes}      # reach
    A = {m: [] for m in modes}      # aim_err
    G = {m: [] for m in modes}      # goal_rate
    rows = [("theta_deg",
             "fixed_aim", "oracle_aim", "rl_aim",
             "fixed_reach", "oracle_reach", "rl_reach",
             "fixed_goal", "oracle_goal", "rl_goal",
             "fixed_lat", "oracle_lat", "rl_lat")]

    for i, th in enumerate(THETAS):
        cseed = 7000 + i * 100
        cell = {}
        for m in modes:
            r = run_cell(m, th, a.N, model, vn, cseed, a.min_reach, a.ang_tol)
            cell[m] = r
            R[m].append(r["reach_mean"]); A[m].append(r["aim_mean"]); G[m].append(r["goal_rate"])

        def g(m, k, d=float("nan")):
            return cell[m][k] if m in cell else d
        rows.append((round(np.degrees(th), 1),
                     g("fixed", "aim_mean"), g("oracle", "aim_mean"), g("rl", "aim_mean"),
                     g("fixed", "reach_mean"), g("oracle", "reach_mean"), g("rl", "reach_mean"),
                     g("fixed", "goal_rate"), g("oracle", "goal_rate"), g("rl", "goal_rate"),
                     g("fixed", "lat_mean"), g("oracle", "lat_mean"), g("rl", "lat_mean")))
        print("θ=%+.2f(%+.0f°) | " % (th, np.degrees(th))
              + " | ".join("%s aim=%.1f reach=%.2f goal=%.2f" %
                           (m, cell[m]["aim_mean"], cell[m]["reach_mean"], cell[m]["goal_rate"])
                           for m in modes))

    csv_name = f"eval_{a.tag}_final.csv"
    with open(csv_name, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("\n[mean over theta grid]")
    for m in modes:
        print("  %-7s | aim=%.1f°  reach=%.2fm  goal=%.2f" %
              (m, np.mean(A[m]), np.mean(R[m]), np.mean(G[m])))

    # ── figures ──────────────────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xdeg = [np.degrees(t) for t in THETAS]
        label = {"fixed": "Fixed analytic (single design)",
                 "oracle": "Oracle analytic (per-goal)",
                 "rl": "RL v5-A (learned)"}
        style = {"fixed": "o--", "oracle": "^--", "rl": "s-"}

        # ① 조준 정확도 (낮을수록 정밀)
        plt.figure(figsize=(6.2, 4.2))
        for m in modes:
            plt.plot(xdeg, A[m], style[m], label=label[m])
        plt.xlabel("commanded direction theta (deg)"); plt.ylabel("aim error (deg)")
        plt.title("Aiming accuracy — classical precise vs learned coarse")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"aim_err_{a.tag}.png", dpi=150); plt.close()

        # ② 킥 파워 (높을수록 강력)
        plt.figure(figsize=(6.2, 4.2))
        for m in modes:
            plt.plot(xdeg, R[m], style[m], label=label[m])
        plt.xlabel("commanded direction theta (deg)"); plt.ylabel("reach distance (m)")
        plt.title("Kick power — learned ~4x stronger than classical")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"reach_{a.tag}.png", dpi=150); plt.close()

        # ③ 방향 골 성공률 (헤드라인)
        plt.figure(figsize=(6.2, 4.2))
        for m in modes:
            plt.plot(xdeg, G[m], style[m], label=label[m])
        plt.xlabel("commanded direction theta (deg)"); plt.ylabel("direction goal success rate")
        plt.title("Directional goal success (aim_err < %.0f deg)" % a.ang_tol)
        plt.ylim(-0.05, 1.05); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"goal_dir_{a.tag}.png", dpi=150); plt.close()

        print("saved %s, aim_err_%s.png, reach_%s.png, goal_dir_%s.png"
              % (csv_name, a.tag, a.tag, a.tag))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
