# train_v55.py — v5.5: v5-A 베이스 + 조준 정렬 보상 강화(w_align 0.5→1.2)
# 변경점은 w_align 하나뿐. 나머지는 train_v5(v5-A)와 동일.
import argparse, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from env_v5 import OP3KickEnvV5


def make_env(kick_window, w_align, w_height, w_squash):
    def _f():
        return OP3KickEnvV5(kick_window=kick_window, max_steps=300,
                            aim_range=0.4, d_min=1.0, d_max=1.8,
                            w_align=w_align,                        # ★ v5.5 핵심: 조준 강화
                            w_height=w_height, w_squash=w_squash,
                            full_action=False,                      # 마스킹(v5-A 그대로)
                            obs_noise_std=0.01, act_latency=1)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--n_envs", type=int, default=8); ap.add_argument("--kick_window", type=int, default=12)
    ap.add_argument("--w_align", type=float, default=1.2)         # ★ v5-A는 0.5 → v5.5는 1.2
    ap.add_argument("--w_height", type=float, default=0.0)
    ap.add_argument("--w_squash", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent_coef", type=float, default=0.005)
    ap.add_argument("--out", type=str, default="runs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs("logs", exist_ok=True)
    venv = VecNormalize(SubprocVecEnv([make_env(a.kick_window, a.w_align, a.w_height, a.w_squash)
                                       for _ in range(a.n_envs)]),
                        norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", venv, device="cpu", verbose=1, seed=a.seed, tensorboard_log="./tb",
                n_steps=2048, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                ent_coef=a.ent_coef, learning_rate=a.lr, target_kl=0.05,
                policy_kwargs=dict(net_arch=[256, 256]))
    ckpt = CheckpointCallback(save_freq=max(50_000 // a.n_envs, 1), save_path=a.out,
                              name_prefix=f"op3_kick_v55_s{a.seed}")
    model.learn(total_timesteps=a.steps, callback=ckpt)
    model.save(os.path.join(a.out, f"op3_kick_v55_s{a.seed}"))
    venv.save(os.path.join(a.out, f"vecnorm_v55_s{a.seed}.pkl"))
    print("saved:", a.out)


if __name__ == "__main__":
    main()