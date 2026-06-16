# train_v5b.py  [미검증 — viewer로 지지각 확정 후 실행]
# v5-B = 물리적 고정(env_v5b: 고정 관절 매 스텝 qpos 강제) + 왼다리 지지자세. 오른다리만 정책 구동.
# 안정화(lr=1e-4·ent_coef=0.005·target_kl=0.05)는 v5-A에서 검증됨(approx_kl 0.92→0.043) → 유지.
# ★선행: env_v5b.py의 HOLD_POSE(왼다리 l_hip_pitch/l_knee/l_ank_pitch)를 viewer 실측값으로 교체.
# 실행: python train_v5b.py --seed 0 --steps 5000000 --n_envs 8 --kick_window 12 > logs\v5b_s0.log 2>&1
import argparse, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from env_v5b import OP3KickEnvV5B


def make_env(kick_window, w_height, w_squash):
    def _f():
        return OP3KickEnvV5B(kick_window=kick_window, max_steps=300,
                             aim_range=0.4, d_min=1.0, d_max=1.8,
                             w_height=w_height, w_squash=w_squash,
                             full_action=False,                     # 마스킹+물리고정(학습)
                             obs_noise_std=0.01, act_latency=1)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--n_envs", type=int, default=8); ap.add_argument("--kick_window", type=int, default=12)
    ap.add_argument("--w_height", type=float, default=0.0)        # 깔기 잔존 시 0.2~0.5
    ap.add_argument("--w_squash", type=float, default=0.0)        # 깔기 잔존 시 0.3~0.8
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent_coef", type=float, default=0.005)     # std<0.05 조기붕괴 시 ↑
    ap.add_argument("--out", type=str, default="runs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs("logs", exist_ok=True)
    venv = VecNormalize(SubprocVecEnv([make_env(a.kick_window, a.w_height, a.w_squash) for _ in range(a.n_envs)]),
                        norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", venv, device="cpu", verbose=1, seed=a.seed, tensorboard_log="./tb",
                n_steps=2048, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                ent_coef=a.ent_coef, learning_rate=a.lr, target_kl=0.05,
                policy_kwargs=dict(net_arch=[256, 256]))
    ckpt = CheckpointCallback(save_freq=max(50_000 // a.n_envs, 1), save_path=a.out, name_prefix=f"op3_kick_v5b_s{a.seed}")
    model.learn(total_timesteps=a.steps, callback=ckpt)
    model.save(os.path.join(a.out, f"op3_kick_v5b_s{a.seed}")); venv.save(os.path.join(a.out, f"vecnorm_v5b_s{a.seed}.pkl"))
    print("saved:", a.out)


if __name__ == "__main__":
    main()
