# train.py  v2  [패턴 검증완료 2026-06-15: PPO.learn PASS]
# OP3 차기 PPO 학습(CPU). 학습 중 DR 수준 0~3 무작위 혼합. 시드별 + 체크포인트.
# 직진 차기(헤드라인): --goal_cond 0 (기본).  조준 과제(L3): --goal_cond 1.
# 실행: python train.py --seed 0 --steps 5000000 --n_envs 8 > logs/seed0.log 2>&1
import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from op3_kick_env import OP3KickEnv
from domain_randomization import DomainRandomize


def make_env(goal_cond):
    def _f():
        lvl = int(np.random.choice([0, 1, 2, 3]))
        return DomainRandomize(OP3KickEnv(max_steps=200, goal_cond=goal_cond), level=lvl)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--n_envs", type=int, default=8)        # 실코어 수에 맞춰(os.cpu_count())
    ap.add_argument("--goal_cond", type=int, default=0)     # 0=직진(헤드라인), 1=조준(L3)
    ap.add_argument("--out", type=str, default="runs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); os.makedirs("logs", exist_ok=True)

    venv = SubprocVecEnv([make_env(bool(args.goal_cond)) for _ in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
    model = PPO(
        "MlpPolicy", venv, device="cpu", verbose=1, seed=args.seed, tensorboard_log="./tb",
        n_steps=2048, batch_size=512, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.0, learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    tag = f"s{args.seed}_g{args.goal_cond}"
    ckpt = CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1), save_path=args.out,
                              name_prefix=f"op3_kick_{tag}")
    model.learn(total_timesteps=args.steps, callback=ckpt)
    model.save(os.path.join(args.out, f"op3_kick_ppo_{tag}"))
    venv.save(os.path.join(args.out, f"vecnorm_{tag}.pkl"))
    print("saved:", args.out, tag)


if __name__ == "__main__":
    main()
