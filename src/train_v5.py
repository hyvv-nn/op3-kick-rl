# train_v5.py  [미검증 — 다음 세션 실행]
# v5 = 액션 마스킹(차는 다리만) + (선택)높이/중심타격 보상 + 학습 안정화. flailing 미해결 해결이 목적.
# v5-A(권장 1차): 액션마스킹만. python train_v5.py --seed 0 --steps 5000000 --n_envs 8 --kick_window 12 > logs\v5_s0.log 2>&1
# v5-B(깔기 지속 시): 보상 추가. ... --w_height 0.3 --w_squash 0.5
import argparse, os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from env_v5 import OP3KickEnvV5


def make_env(kick_window, w_height, w_squash):
    def _f():
        return OP3KickEnvV5(kick_window=kick_window, max_steps=300,
                            aim_range=0.4, d_min=1.0, d_max=1.8,
                            w_height=w_height, w_squash=w_squash,
                            full_action=False,                      # ★ 마스킹(차는 다리만)
                            obs_noise_std=0.01, act_latency=1)      # 약한 DR
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--n_envs", type=int, default=8); ap.add_argument("--kick_window", type=int, default=12)
    ap.add_argument("--w_height", type=float, default=0.0)         # v5-A 기본 0 / v5-B에서 0.2~0.5
    ap.add_argument("--w_squash", type=float, default=0.0)         # v5-A 기본 0 / v5-B에서 0.3~0.8
    ap.add_argument("--lr", type=float, default=1e-4)             # v4 approx_kl 높음 → 3e-4에서 ↓
    ap.add_argument("--ent_coef", type=float, default=0.005)     # v4 std 조기붕괴(0.085) → 탐색 유지
    ap.add_argument("--out", type=str, default="runs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs("logs", exist_ok=True)
    venv = VecNormalize(SubprocVecEnv([make_env(a.kick_window, a.w_height, a.w_squash) for _ in range(a.n_envs)]),
                        norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", venv, device="cpu", verbose=1, seed=a.seed, tensorboard_log="./tb",
                n_steps=2048, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                ent_coef=a.ent_coef, learning_rate=a.lr, target_kl=0.05,   # target_kl로 과대갱신 제동
                policy_kwargs=dict(net_arch=[256, 256]))
    ckpt = CheckpointCallback(save_freq=max(50_000 // a.n_envs, 1), save_path=a.out, name_prefix=f"op3_kick_v5_s{a.seed}")
    model.learn(total_timesteps=a.steps, callback=ckpt)
    model.save(os.path.join(a.out, f"op3_kick_v5_s{a.seed}")); venv.save(os.path.join(a.out, f"vecnorm_v5_s{a.seed}.pkl"))
    print("saved:", a.out)


if __name__ == "__main__":
    main()
