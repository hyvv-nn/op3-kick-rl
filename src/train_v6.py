# train_v6.py  [B(골대 조준) 학습 — 2026-06-17 Cowork 사전작성. 실학습은 사용자 터미널(밤샘)]
# ─────────────────────────────────────────────────────────────────────────────
# train_v5 기반. env_v6 단계적 학습. 계획서 B-2 준수: 한 번에 다 넣지 말 것.
#
# 권장 순서 (warm-start 우선):
#   0) python warmstart_v6.py --src runs\op3_kick_v5_s0.zip --srcvec runs\vecnorm_v5_s0.pkl --seed 0 --check
#   1) Stage1 (골대 인식, 보상=v5-A): "골대 봐도 성능 유지" 확인
#      python train_v6.py --stage 1 --warmstart --seed 0 --steps 2000000 --n_envs 8 > logs\v6s1.log 2>&1
#   2) Stage2 (골 보상): 골 성공률 학습
#      python train_v6.py --stage 2 --warmstart_from runs\op3_kick_v6_s1.zip --w_goal 5.0 --seed 0 --steps 4000000 ...
#   3) (선택) Stage3 안정화: --w_arate 0.15 --w_settle 0.05 --kick_window 0  (0=종료없음=킥후까지 학습)
# fresh(처음부터): --warmstart 빼면 무작위 초기화.
# ─────────────────────────────────────────────────────────────────────────────
import argparse, os, pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from env_v6 import OP3KickEnvV6


def make_env(stage, w_goal, w_vel, w_arate, w_settle, settle_after, kick_window, goal_randomize, aim_range, d_min, d_max):
    goal_as_target = (stage >= 2)
    def _f():
        return OP3KickEnvV6(kick_window=kick_window, max_steps=300,
                            aim_range=aim_range, d_min=d_min, d_max=d_max,
                            w_arate=w_arate, w_goal=(w_goal if stage >= 2 else 0.0),
                            w_vel=w_vel, w_settle=w_settle, settle_after=settle_after,
                            goal_as_target=goal_as_target, goal_randomize=goal_randomize,
                            full_action=False, obs_noise_std=0.01, act_latency=1)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--kick_window", type=int, default=12, help="0=종료없음(킥 이후까지 학습, settle용)")
    ap.add_argument("--w_goal", type=float, default=5.0)
    ap.add_argument("--w_vel", type=float, default=0.0, help="파워: 공 속도 보상(0.5~1.0). 약함/느림 대응")
    ap.add_argument("--aim_range", type=float, default=0.4, help="학습 방향범위 rad(0.4=±23°, 0.8=±45° 광각)")
    ap.add_argument("--d_min", type=float, default=1.0); ap.add_argument("--d_max", type=float, default=1.8, help="표적거리 범위(파워: d_max 3.0)")
    ap.add_argument("--tag", type=str, default="", help="출력 모델명 suffix(미지정시 v6_s{seed}). config별 충돌 방지")
    ap.add_argument("--settle_after", type=int, default=20, help="이 스텝 이후 전체관절 정적 페널티(w_settle>0, kick_window=0과)")
    ap.add_argument("--w_arate", type=float, default=0.05, help="jerk↓: 0.1~0.2로 올리면 명령 매끄러움↑")
    ap.add_argument("--w_settle", type=float, default=0.0, help=">0: 킥후 비킥관절 정적 보상(Stage3)")
    ap.add_argument("--goal_randomize", action="store_true")
    ap.add_argument("--warmstart", action="store_true", help="runs/op3_kick_v6_init.* 에서 이어 학습")
    ap.add_argument("--warmstart_from", type=str, default="", help="특정 zip에서 이어 학습(vecnorm 동일 stem .pkl 가정)")
    ap.add_argument("--smoke", action="store_true", help="짧은 구성 점검(DummyVecEnv 1개)")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs("logs", exist_ok=True)

    kw = None if a.kick_window == 0 else a.kick_window
    n_envs = 1 if a.smoke else a.n_envs
    Vec = DummyVecEnv if a.smoke else SubprocVecEnv
    base = Vec([make_env(a.stage, a.w_goal, a.w_vel, a.w_arate, a.w_settle, a.settle_after, kw, a.goal_randomize, a.aim_range, a.d_min, a.d_max) for _ in range(n_envs)])

    init_model = ""
    init_vec = ""
    if a.warmstart_from:
        init_model = a.warmstart_from
        init_vec = os.path.splitext(a.warmstart_from)[0].replace("op3_kick", "vecnorm") + ".pkl"
    elif a.warmstart:
        init_model = os.path.join(a.out, "op3_kick_v6_init.zip")
        init_vec = os.path.join(a.out, "vecnorm_v6_init.pkl")

    if init_model and os.path.exists(init_model):
        venv = VecNormalize(base, norm_obs=True, norm_reward=True)
        if os.path.exists(init_vec):
            vn = pickle.load(open(init_vec, "rb"))
            venv.obs_rms = vn.obs_rms; venv.ret_rms = vn.ret_rms
        model = PPO.load(init_model, env=venv, device="cpu")
        print("[warm-start] from", init_model)
    else:
        venv = VecNormalize(base, norm_obs=True, norm_reward=True)
        model = PPO("MlpPolicy", venv, device="cpu", verbose=0 if a.smoke else 1, seed=a.seed,
                    n_steps=2048, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    ent_coef=0.005, learning_rate=1e-4, target_kl=0.05,
                    policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log=None if a.smoke else "./tb")
        print("[fresh] random init")

    if a.smoke:
        model.learn(total_timesteps=4096)
        print("[smoke] OK — learn() ran, obs_dim=", venv.observation_space.shape)
        return

    tag = a.tag if a.tag else f"v6_s{a.seed}"
    ckpt = CheckpointCallback(save_freq=max(100_000 // n_envs, 1), save_path=a.out,
                              name_prefix=f"op3_kick_{tag}")
    model.learn(total_timesteps=a.steps, callback=ckpt)
    model.save(os.path.join(a.out, f"op3_kick_{tag}"))
    venv.save(os.path.join(a.out, f"vecnorm_{tag}.pkl"))
    print("[saved] runs/op3_kick_%s.zip + vecnorm_%s.pkl" % (tag, tag))
    print("saved:", a.out)


if __name__ == "__main__":
    main()
