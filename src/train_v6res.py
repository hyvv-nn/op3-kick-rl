# train_v6res.py — Residual 하이브리드 학습 (env_v6res). v6clean에서 warm-start.
#   train_v6의 PPO 구성과 동일(net[256,256], lr1e-4, target_kl0.05 등은 warm-start 모델에서 승계).
#   차이: env가 OP3KickEnvV6Res(analytic baseline + 잔차). 보상/obs는 env_v6와 동일.
# 실행:
#   python train_v6res.py --warmstart_from runs\op3_kick_v6clean.zip --res_scale 1.0 --tag v6res --seed 0 --steps 5000000 --n_envs 8
#   스모크: python train_v6res.py --smoke
import argparse, os, pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from env_v6res import OP3KickEnvV6Res


def make_env(res_scale, w_vel, w_arate, w_settle, settle_after, aim_range, d_min, d_max):
    def _f():
        return OP3KickEnvV6Res(kick_window=None, max_steps=300, aim_range=aim_range,
                               d_min=d_min, d_max=d_max, w_arate=w_arate, w_goal=0.0,
                               w_vel=w_vel, w_settle=w_settle, settle_after=settle_after,
                               goal_as_target=False, goal_randomize=False,
                               obs_noise_std=0.01, act_latency=1, res_scale=res_scale)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart_from", default="runs/op3_kick_v6clean.zip")
    ap.add_argument("--res_scale", type=float, default=1.0, help="잔차 권한(1.0=완전, <1=보정 위주)")
    ap.add_argument("--w_vel", type=float, default=0.8)
    ap.add_argument("--w_arate", type=float, default=0.12)
    ap.add_argument("--w_settle", type=float, default=0.05)
    ap.add_argument("--settle_after", type=int, default=25)
    ap.add_argument("--aim_range", type=float, default=0.8)
    ap.add_argument("--d_min", type=float, default=1.5)
    ap.add_argument("--d_max", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--tag", default="v6res")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    n = 1 if a.smoke else a.n_envs
    Vec = DummyVecEnv if a.smoke else SubprocVecEnv
    base = Vec([make_env(a.res_scale, a.w_vel, a.w_arate, a.w_settle, a.settle_after,
                         a.aim_range, a.d_min, a.d_max) for _ in range(n)])
    venv = VecNormalize(base, norm_obs=True, norm_reward=True)

    init_vec = os.path.splitext(a.warmstart_from)[0].replace("op3_kick", "vecnorm") + ".pkl"
    if os.path.exists(a.warmstart_from):
        if os.path.exists(init_vec):
            vn = pickle.load(open(init_vec, "rb")); venv.obs_rms = vn.obs_rms; venv.ret_rms = vn.ret_rms
        model = PPO.load(a.warmstart_from, env=venv, device="cpu")
        print("[warm-start]", a.warmstart_from, "res_scale=", a.res_scale)
    else:
        model = PPO("MlpPolicy", venv, device="cpu", verbose=1, seed=a.seed, n_steps=2048,
                    batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
                    learning_rate=1e-4, target_kl=0.05, policy_kwargs=dict(net_arch=[256, 256]),
                    tensorboard_log="./tb")
        print("[fresh] (warmstart 파일 없음) res_scale=", a.res_scale)

    if a.smoke:
        model.learn(total_timesteps=4096)
        print("[smoke] OK obs", venv.observation_space.shape, "act", venv.action_space.shape); return
    ckpt = CheckpointCallback(save_freq=max(100_000 // n, 1), save_path=a.out, name_prefix=f"op3_kick_{a.tag}")
    model.learn(total_timesteps=a.steps, callback=ckpt)
    model.save(os.path.join(a.out, f"op3_kick_{a.tag}")); venv.save(os.path.join(a.out, f"vecnorm_{a.tag}.pkl"))
    print("[saved] runs/op3_kick_%s.zip + vecnorm_%s.pkl" % (a.tag, a.tag))


if __name__ == "__main__":
    main()
