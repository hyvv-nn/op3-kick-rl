# train.py  v1 [초기 버전. 정식=상위 폴더 v2(--goal_cond 지원)]
# OP3 직진 차기 PPO 학습(CPU). 학습 중 DR 수준 0~3 무작위 혼합. 시드별 + 체크포인트.
# 실행: python train.py --seed 0 --steps 5000000 --n_envs 8 > logs/seed0.log 2>&1
import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from op3_kick_env import OP3KickEnv
from domain_randomization import DomainRandomize


def make_env():
    def _f():
        lvl = int(np.random.choice([0, 1, 2, 3]))
        return DomainRandomize(OP3KickEnv(max_steps=200), level=lvl)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--out", type=str, default="runs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); os.makedirs("logs", exist_ok=True)

    venv = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
    model = PPO(
        "MlpPolicy", venv, device="cpu", verbose=1, seed=args.seed, tensorboard_log="./tb",
        n_steps=2048, batch_size=512, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.0, learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    ckpt = CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1), save_path=args.out,
                              name_prefix=f"op3_kick_s{args.seed}")
    model.learn(total_timesteps=args.steps, callback=ckpt)
    model.save(os.path.join(args.out, f"op3_kick_ppo_s{args.seed}"))
    venv.save(os.path.join(args.out, f"vecnorm_s{args.seed}.pkl"))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
