# op3_kick_env.py  v1 [초기 버전. 정식=상위 폴더 v2]
# 고정베이스(골반 코드 핀) OP3 직진 차기. 목표 고정(+x). goal-cond/노이즈/지연/접촉보너스 없음.
# obs(48)=joint(40)+ball(6)+target_dir(2), action(20). 보상=2*(목표방향 공속도)+5*(전진)-0.001||a||^2.
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))


class OP3KickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 40}

    def __init__(self, scene=None, pin_z=0.33, frame_skip=5, max_steps=200, render_mode=None):
        scene = scene or os.path.join(HERE, "op3_kick_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.pin_z = pin_z
        self._root = np.array([0, 0, pin_z, 1, 0, 0, 0.0])
        m = self.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.bq = m.jnt_qposadr[jid]
        self.bv = m.jnt_dofadr[jid]
        self.foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_ank_roll_link")
        self.target = np.array([1.2, -0.07])
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
        self.action_space = spaces.Box(-1.0, 1.0, (m.nu,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (20 + 20 + 3 + 3 + 2,), np.float32)
        self._renderer = None
        self._step = 0

    def _pin(self):
        self.data.qpos[0:7] = self._root
        self.data.qvel[0:6] = 0.0

    def _obs(self):
        d = self.data
        jq = d.qpos[7:27]
        jv = d.qvel[6:26]
        ball = d.qpos[self.bq:self.bq + 3]
        ballv = d.qvel[self.bv:self.bv + 3]
        tdir = self.target - ball[:2]
        tdir = tdir / (np.linalg.norm(tdir) + 1e-8)
        return np.concatenate([jq, jv, ball, ballv, tdir]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._pin(); mujoco.mj_forward(self.model, self.data)
        fx, fy, _ = self.data.xpos[self.foot]
        dx = getattr(self, "_dx", 0.0)
        self.data.qpos[self.bq:self.bq + 3] = [fx + 0.05 + dx, fy, 0.05]
        self.data.qpos[self.bq + 3:self.bq + 7] = [1, 0, 0, 0]
        self.data.qvel[self.bv:self.bv + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        ctrl = self.ctrl_lo + (a + 1.0) * 0.5 * (self.ctrl_hi - self.ctrl_lo)
        b0 = self.data.qpos[self.bq:self.bq + 2].copy()
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data); self._pin()
        mujoco.mj_forward(self.model, self.data)
        d = self.data
        ball = d.qpos[self.bq:self.bq + 2]
        ballv = d.qvel[self.bv:self.bv + 2]
        to = self.target - ball
        to = to / (np.linalg.norm(to) + 1e-8)
        reward = (2.0 * float(np.dot(ballv, to)) + 5.0 * float(np.dot(ball - b0, to))
                  - 0.001 * float(np.sum(a ** 2)))
        self._step += 1
        truncated = self._step >= self.max_steps
        info = {"ball_x": float(ball[0]), "ball_disp": float(np.linalg.norm(ball - b0))}
        return self._obs(), reward, False, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
