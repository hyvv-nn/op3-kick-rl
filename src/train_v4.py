# train_v4.py  [미검증 — 다음 세션 실행]
# v4 목표조건부(θ,D) + 흔들기 처벌 + 임펄스 종료(kick_window) 학습. 흐느적댐 해결이 목적.
# 실행: python train_v4.py --seed 0 --steps 5000000 --n_envs 8 --kick_window 8 > logs\v4_s0.log 2>&1
import argparse, os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from env_v4 import OP3KickEnvV4


def make_env(kick_window):
    def _f():
        return OP3KickEnvV4(kick_window=kick_window, max_steps=300,
                            aim_range=0.4, d_min=1.0, d_max=1.8,
                            obs_noise_std=0.01, act_latency=1)   # 약한 DR 동반(강건성)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--n_envs", type=int, default=8); ap.add_argument("--kick_window", type=int, default=8)
    ap.add_argument("--out", type=str, default="runs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs("logs", exist_ok=True)
    venv = VecNormalize(SubprocVecEnv([make_env(a.kick_window) for _ in range(a.n_envs)]),
                        norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", venv, device="cpu", verbose=1, seed=a.seed, tensorboard_log="./tb",
                n_steps=2048, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                ent_coef=0.0, learning_rate=3e-4, policy_kwargs=dict(net_arch=[256, 256]))
    ckpt = CheckpointCallback(save_freq=max(50_000 // a.n_envs, 1), save_path=a.out, name_prefix=f"op3_kick_v4_s{a.seed}")
    model.learn(total_timesteps=a.steps, callback=ckpt)
    model.save(os.path.join(a.out, f"op3_kick_v4_s{a.seed}")); venv.save(os.path.join(a.out, f"vecnorm_v4_s{a.seed}.pkl"))
    print("saved:", a.out)


if __name__ == "__main__":
    main()
