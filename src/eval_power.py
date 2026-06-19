# eval_power.py — 정직한 파워 측정: 공 launch/peak 속도(m/s). reach(굴림 거품) 대신.
# ─────────────────────────────────────────────────────────────────────────────
# 왜: reach(~6~7m)는 바닥마찰 ~0 + 경량공(0.15kg)의 과대평가. 진짜 파워 = 공 peak 속도.
# 벤치마크(동일 OP3): DeepMind planted 2.0 / run-up 2.8~3.4 m/s (Science Robotics 2024).
# 순수 RL(v6clean/goalb/v7) -> env_v6. 잔차(v6res*) -> --residual. Stage-2/goal -> --goal_cond.
# (옵션) --ball_mass 0.43 : FIFA 질량 공으로도 1점 보고(질량·관성 스케일, [근사]).
# 실행:
#   python eval_power.py --model runs\op3_kick_v6goalb_s2.zip --vecnorm runs\vecnorm_v6goalb_s2.pkl --goal_cond --N 10 --tag v6goalb_s2
# ─────────────────────────────────────────────────────────────────────────────
import argparse, csv
import numpy as np

THETAS = [-0.4, -0.2, 0.0, 0.2, 0.4]   # 명령 방향(rad). 정면 위주(파워는 방향 둔감).
D_FIX = 2.0
MAXSTEPS = 200


def _norm(o, vn):
    if vn is None:
        return o
    return np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                   -vn.clip_obs, vn.clip_obs).astype(np.float32)


def _set_ball_mass(u, m):
    if m <= 0:
        return None
    gid = u.ball_gid
    bid = int(u.model.geom_bodyid[gid])
    old = float(u.model.body_mass[bid])
    if old > 0:
        r = m / old
        u.model.body_mass[bid] = m
        u.model.body_inertia[bid] = u.model.body_inertia[bid] * r   # 관성도 비례 스케일[근사]
    return old


def run_cell(th, N, model, vn, seed0, residual, res_scale, goal_cond, ball_mass):
    if residual:
        from env_v6res import OP3KickEnvV6Res
        env = OP3KickEnvV6Res(kick_window=None, max_steps=MAXSTEPS,
                              aim_range=abs(th) + 1e-3, d_min=D_FIX, d_max=D_FIX, res_scale=res_scale)
    else:
        from env_v6 import OP3KickEnvV6
        env = OP3KickEnvV6(kick_window=None, max_steps=MAXSTEPS,
                           aim_range=abs(th) + 1e-3, d_min=D_FIX, d_max=D_FIX, full_action=False)
    u = env.unwrapped
    _set_ball_mass(u, ball_mass)
    peaks = []
    for s in range(N):
        env.reset(seed=seed0 + s); env.reset_bonus()
        u.theta, u.D = th, D_FIX
        pt = u.ball0 + D_FIX * np.array([np.cos(th), np.sin(th)])
        u.target = pt
        if goal_cond:
            u.goal_pos = pt.astype(np.float32).copy()
        u._prevdist = float(np.linalg.norm(u.target - u.ball0))
        pk = 0.0
        for _ in range(MAXSTEPS):
            o = u._obs()
            a, _ = model.predict(_norm(o, vn), deterministic=True)
            env.step(a)
            v = float(np.linalg.norm(u.data.qvel[u.bv:u.bv + 2]))   # 공 평면속도
            if v > pk:
                pk = v
        peaks.append(pk)
    peaks = np.array(peaks)
    return float(peaks.mean()), float(peaks.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--res_scale", type=float, default=0.5)
    ap.add_argument("--goal_cond", action="store_true")
    ap.add_argument("--ball_mass", type=float, default=0.0, help=">0이면 공 질량 설정(FIFA 0.43). 기본=씬값(~0.15)")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--tag", default="model")
    a = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(a.model, device="cpu")
    vn = None
    if a.vecnorm:
        import pickle
        vn = pickle.load(open(a.vecnorm, "rb"))

    rows = [("theta_deg", "peak_mean_mps", "peak_std")]
    allpk = []
    for th in THETAS:
        m, s = run_cell(th, a.N, model, vn, 9000, a.residual, a.res_scale, a.goal_cond, a.ball_mass)
        rows.append((round(np.degrees(th), 1), round(m, 3), round(s, 3)))
        allpk.append(m)
        print("theta=%+.0f deg | peak ball speed = %.3f m/s" % (np.degrees(th), m))
    overall = float(np.mean(allpk))
    csv_name = f"power_{a.tag}.csv"
    with open(csv_name, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    bm = a.ball_mass if a.ball_mass > 0 else 0.15
    print("\n[%s] overall peak ball speed = %.3f m/s  (ball ~ %.2f kg)" % (a.tag, overall, bm))
    print("  benchmark (same OP3, DeepMind 2024): planted 2.0 / run-up 2.8~3.4 m/s")
    print("  saved", csv_name)


if __name__ == "__main__":
    main()
