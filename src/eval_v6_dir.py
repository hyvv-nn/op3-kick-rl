# eval_v6_dir.py  [env_v6(obs50) 모델의 방향(theta) eval — 위치(qpos) 기반, 속도 미사용]
# ─────────────────────────────────────────────────────────────────────────────
# 목적: eval_v5a_final.py와 *완전히 동일한 방향 지표*(reach·aim_err·directional goal)를
#       env_v6(obs50, goal_rel 포함) 모델에 적용. v5-A 헤드라인과 같은 theta축에서 직접 비교.
#       기존 eval_v6.py(goal_y sweep)의 한계를 보완 — eval_v6.py는 골 판정이 (1) 고정중심
#       (-0.07) 기준 (2) 골폭 ±0.6m가 sweep ±0.45m보다 넓어, 전 config·전 y에서 1.0 포화.
#       본 eval은 *거리무관 각도기준* directional goal로, 광각(±45°)까지 판별력이 있다.
#
# in-distribution 보장: v6clean은 --stage 1 학습(goal_as_target=False) →
#   학습 시 target=랜덤(theta,D) 방향점, goal_pos=고정 goal_def.goal_pos()=(1.5,-0.07)(goal_randomize 미사용).
#   따라서 본 eval은 goal_pos를 *건드리지 않고*(reset의 _place_goal이 (1.5,-0.07)로 설정) target만 theta로 지정.
#
# 3지표(전부 위치 변위 disp = qpos - ball0 에서 유도, eval_v5a_final과 동일 정의):
#   (1) aim_err(deg) = |실제방향 - 명령 theta|            ← 조준 정확도
#   (2) reach(m)     = proj(명령방향 투영거리)             ← 킥 파워
#   (3) goal_dir 0/1 = (proj >= MIN_REACH) and (aim_err < ANG_TOL)  ← 방향 골(거리무관)
#
# 비교군: fixed(단일설계 analytic) / oracle(theta별 재설계 analytic) / rl(v6 모델, 6관절 마스킹).
#
# 실행:
#   python eval_v6_dir.py --model runs\op3_kick_v6clean.zip --vecnorm runs\vecnorm_v6clean.pkl --N 20 --tag v6clean_s0
#   (3-seed: s1,s2도 동일하게 --model ..._s1/_s2.zip --vecnorm ..._s1/_s2.pkl --tag v6clean_s1/_s2)
# 산출: eval_<tag>_dir.csv + aim_err_<tag>.png + reach_<tag>.png + goal_dir_<tag>.png (cwd)
# ─────────────────────────────────────────────────────────────────────────────
import argparse, csv
import numpy as np
from env_v6 import OP3KickEnvV6
from analytic_kick import make_kick
import goal_def

# theta 격자(rad): v5-A 학습범위 ±0.4(±23°) + v6clean 광각 학습범위 ±0.8(±45°)까지 확장.
THETAS = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
D_FIX   = 2.0       # 명령 거리(m). v6clean 학습 d∈[1.5,3.0] 내. 방향골은 거리무관이라 D 영향 최소.
MAXSTEPS = 200      # 위치 기반 → coast 무관, 200스텝이면 충분.
DT = 0.025          # env frame_skip(5)*timestep(0.005) 정합.
D_REF = 1.4
BASE = dict(amp_knee=0.2, t_swing=0.15)

MIN_REACH_DEFAULT = 0.6     # m   : 골 인정 최소 전진(헛발질 배제).
ANG_TOL_DEFAULT   = 15.0    # deg : 방향 골 허용오차(거리무관). eval_v5a_final과 동일.


def power_for_D(D):
    return float(np.clip(D / D_REF, 0.5, 2.0))


def _norm(o, vn):
    if vn is None:
        return o
    return np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                   -vn.clip_obs, vn.clip_obs).astype(np.float32)


def _metrics(u, th):
    """위치 변위 기반 (proj, aim_err_deg, lateral). 속도 미사용 (eval_v5a_final 동일)."""
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


def _set_target(u, th, D, goal_cond=False):
    """target을 (theta,D) 방향점으로. goal_cond=True면 goal_pos도 같은 점으로 이동
    (Stage-2/goal-conditioned 모델: 학습 때 target=goal 이었으므로 in-distribution 평가)."""
    u.theta, u.D = th, D
    pt = u.ball0 + D * np.array([np.cos(th), np.sin(th)])
    u.target = pt
    if goal_cond:
        u.goal_pos = pt.astype(np.float32).copy()
    u._prevdist = float(np.linalg.norm(u.target - u.ball0))


def run_cell(mode, th, N, model, vn, cell_seed, min_reach, ang_tol, residual=False, res_scale=1.0, goal_cond=False):
    full = (mode != "rl")
    if mode == "rl" and residual:
        from env_v6res import OP3KickEnvV6Res
        env = OP3KickEnvV6Res(kick_window=None, max_steps=MAXSTEPS,
                              aim_range=abs(th) + 1e-3, d_min=D_FIX, d_max=D_FIX, res_scale=res_scale)
    else:
        env = OP3KickEnvV6(kick_window=None, max_steps=MAXSTEPS,
                           aim_range=abs(th) + 1e-3, d_min=D_FIX, d_max=D_FIX, full_action=full)
    u = env.unwrapped
    PR, AE, LA = [], [], []
    for s in range(N):
        env.reset(seed=cell_seed + s); env.reset_bonus()
        _set_target(u, th, D_FIX, goal_cond)
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
    goal = (PR >= min_reach) & (AE < ang_tol)       # 거리무관 각도 기준
    return dict(reach_mean=float(PR.mean()), reach_std=float(PR.std()),
                aim_mean=float(AE.mean()), aim_std=float(AE.std()),
                lat_mean=float(LA.mean()),
                goal_rate=float(goal.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--vecnorm")
    ap.add_argument("--no_analytic", action="store_true", help="analytic 비교군 생략(rl만)")
    ap.add_argument("--residual", action="store_true", help="RL 모델이 env_v6res(잔차) 학습본일 때 residual env로 평가")
    ap.add_argument("--res_scale", type=float, default=1.0, help="--residual일 때 학습에 쓴 값과 동일하게")
    ap.add_argument("--goal_cond", action="store_true", help="Stage-2/goal-conditioned 모델: goal_pos를 theta 방향으로 이동해 평가")
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--min_reach", type=float, default=MIN_REACH_DEFAULT)
    ap.add_argument("--ang_tol", type=float, default=ANG_TOL_DEFAULT)
    ap.add_argument("--tag", type=str, default="v6clean_s0")
    a = ap.parse_args()

    model = vn = None
    if a.model:
        from stable_baselines3 import PPO
        model = PPO.load(a.model, device="cpu")
        if a.vecnorm:
            import pickle
            vn = pickle.load(open(a.vecnorm, "rb"))

    modes = ([] if a.no_analytic else ["fixed", "oracle"]) + (["rl"] if model else [])
    if not modes:
        ap.error("필요: --model (또는 analytic 비교군)")
    R = {m: [] for m in modes}
    A = {m: [] for m in modes}
    G = {m: [] for m in modes}
    Rs = {m: [] for m in modes}     # reach std (3-seed band용)
    As = {m: [] for m in modes}     # aim std
    rows = [("theta_deg",
             *[f"{m}_aim" for m in modes],
             *[f"{m}_reach" for m in modes],
             *[f"{m}_goal" for m in modes],
             *[f"{m}_aim_std" for m in modes],
             *[f"{m}_reach_std" for m in modes])]

    for i, th in enumerate(THETAS):
        cseed = 7000 + i * 100
        cell = {}
        for m in modes:
            r = run_cell(m, th, a.N, model, vn, cseed, a.min_reach, a.ang_tol,
                         residual=a.residual, res_scale=a.res_scale, goal_cond=a.goal_cond)
            cell[m] = r
            R[m].append(r["reach_mean"]); A[m].append(r["aim_mean"]); G[m].append(r["goal_rate"])
            Rs[m].append(r["reach_std"]); As[m].append(r["aim_std"])
        rows.append((round(np.degrees(th), 1),
                     *[round(cell[m]["aim_mean"], 3) for m in modes],
                     *[round(cell[m]["reach_mean"], 3) for m in modes],
                     *[round(cell[m]["goal_rate"], 3) for m in modes],
                     *[round(cell[m]["aim_std"], 3) for m in modes],
                     *[round(cell[m]["reach_std"], 3) for m in modes]))
        print("theta=%+.2f(%+.0f deg) | " % (th, np.degrees(th))
              + " | ".join("%s aim=%.1f reach=%.2f goal=%.2f" %
                           (m, cell[m]["aim_mean"], cell[m]["reach_mean"], cell[m]["goal_rate"])
                           for m in modes))

    csv_name = f"eval_{a.tag}_dir.csv"
    with open(csv_name, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("\n[mean over theta grid]")
    for m in modes:
        print("  %-7s | aim=%.1f deg  reach=%.2f m  goal=%.2f" %
              (m, np.mean(A[m]), np.mean(R[m]), np.mean(G[m])))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xdeg = [np.degrees(t) for t in THETAS]
        label = {"fixed": "Fixed analytic (single design)",
                 "oracle": "Oracle analytic (per-goal)",
                 "rl": f"RL {a.tag} (env_v6)"}
        style = {"fixed": "o--", "oracle": "^--", "rl": "s-"}

        plt.figure(figsize=(6.4, 4.3))
        for m in modes:
            plt.plot(xdeg, A[m], style[m], label=label[m])
        plt.xlabel("commanded direction theta (deg)"); plt.ylabel("aim error (deg)")
        plt.axhline(a.ang_tol, color="gray", ls=":", lw=1, label=f"goal tol {a.ang_tol:.0f} deg")
        plt.title("Aiming accuracy vs commanded direction"); plt.legend(fontsize=8)
        plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"aim_err_{a.tag}.png", dpi=150); plt.close()

        plt.figure(figsize=(6.4, 4.3))
        for m in modes:
            plt.plot(xdeg, R[m], style[m], label=label[m])
        plt.xlabel("commanded direction theta (deg)"); plt.ylabel("reach distance (m)")
        plt.title("Kick power vs commanded direction"); plt.legend(fontsize=8)
        plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"reach_{a.tag}.png", dpi=150); plt.close()

        plt.figure(figsize=(6.4, 4.3))
        for m in modes:
            plt.plot(xdeg, G[m], style[m], label=label[m])
        plt.xlabel("commanded direction theta (deg)"); plt.ylabel("directional goal success rate")
        plt.title("Directional goal success (aim_err < %.0f deg)" % a.ang_tol)
        plt.ylim(-0.05, 1.05); plt.legend(fontsize=8)
        plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"goal_dir_{a.tag}.png", dpi=150); plt.close()

        print("saved %s, aim_err_%s.png, reach_%s.png, goal_dir_%s.png"
              % (csv_name, a.tag, a.tag, a.tag))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
