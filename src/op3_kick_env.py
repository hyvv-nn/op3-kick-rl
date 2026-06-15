# op3_kick_env.py  v2  [샌드박스 검증완료 2026-06-15]
# 고정베이스(골반 코드 핀) OP3 차기 + 디벨롭: 목표조건부(조준)·관측노이즈·액추에이터지연·접촉게이트 보상.
# 실측: nq=27 nv=26 nu=20, 액추 head(2)->arms(6)->legs(12). 전방차기 r_hip_pitch(+)/r_knee(+).
#       조준 r_hip_yaw(idx14) 음(-)이 +theta(+y)로 튼다. 공 spawn offset 0.15(겹침 방지).
import os
import collections
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))


class OP3KickEnv(gym.Env):
    """obs(48)=joint_pos(20)+joint_vel(20)+ball_pos(3)+ball_vel(3)+target_dir(2). action(20)=정규화 목표관절각.

    goal_cond=True 면 매 에피소드 목표 방위각 theta~U(-aim_range,aim_range) 무작위(조준 과제).
    obs_noise_std/act_latency = sim2real 현실성(관측노이즈/액추에이터 지연).
    reward = 2*(목표방향 공속도)+5*(목표방향 전진)-0.001||a||^2 + 0.5*(최초접촉 1회). 결과보상=anti-해킹.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 40}

    def __init__(self, scene=None, pin_z=0.33, frame_skip=5, max_steps=200,
                 goal_cond=True, aim_range=0.35, obs_noise_std=0.0, act_latency=0, render_mode=None):
        scene = scene or os.path.join(HERE, "op3_kick_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.frame_skip, self.max_steps, self.render_mode = frame_skip, max_steps, render_mode
        self.pin_z = pin_z
        self._root = np.array([0, 0, pin_z, 1, 0, 0, 0.0])
        self.goal_cond, self.aim_range = goal_cond, aim_range
        self.obs_noise_std, self.act_latency = obs_noise_std, act_latency
        m = self.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.bq, self.bv = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        self.foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_ank_roll_link")
        self.ball_gid = m.geom("ball_geom").id
        self.floor_gid = m.geom("floor").id
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
        self.action_space = spaces.Box(-1.0, 1.0, (m.nu,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (20 + 20 + 3 + 3 + 2,), np.float32)
        self.target = np.array([1.2, -0.07])
        self.cmd_dir = np.array([1.0, 0.0])
        self._renderer = None
        self._step = 0
        self._abuf = collections.deque()
        self._bonus_done = False

    def _pin(self):
        self.data.qpos[0:7] = self._root
        self.data.qvel[0:6] = 0.0

    def _raw_obs(self):
        d = self.data
        jq = d.qpos[7:27]; jv = d.qvel[6:26]
        ball = d.qpos[self.bq:self.bq + 3]; ballv = d.qvel[self.bv:self.bv + 3]
        tdir = self.target - ball[:2]; tdir = tdir / (np.linalg.norm(tdir) + 1e-8)
        return np.concatenate([jq, jv, ball, ballv, tdir]).astype(np.float32)

    def _obs(self):
        o = self._raw_obs()
        if self.obs_noise_std > 0:
            o = o + np.random.normal(0, self.obs_noise_std, o.shape).astype(np.float32)
        return o

    def _ball_hit(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.ball_gid in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.ball_gid else c.geom1
                if other != self.floor_gid:
                    return True
        return False

    def reset_bonus(self):
        self._bonus_done = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._pin(); mujoco.mj_forward(self.model, self.data)
        fx, fy, _ = self.data.xpos[self.foot]
        dx = getattr(self, "_dx", 0.0)
        self.data.qpos[self.bq:self.bq + 3] = [fx + 0.15 + dx, fy, 0.07]   # offset 0.15 = 발과 겹침 방지(실측)
        self.data.qpos[self.bq + 3:self.bq + 7] = [1, 0, 0, 0]
        self.data.qvel[self.bv:self.bv + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        th = np.random.uniform(-self.aim_range, self.aim_range) if self.goal_cond else 0.0
        self.cmd_dir = np.array([np.cos(th), np.sin(th)])
        D = 1.15
        self.target = np.array([fx + 0.15 + dx + D * np.cos(th), fy + D * np.sin(th)])
        self._ball0 = self.data.qpos[self.bq:self.bq + 2].copy()
        self._step = 0
        self._bonus_done = False
        self._abuf = collections.deque([np.zeros(self.model.nu, np.float32)] * (self.act_latency + 1),
                                       maxlen=self.act_latency + 1)
        return self._obs(), {"cmd_theta": float(th)}

    def step(self, action):
        self._abuf.append(np.clip(np.asarray(action, np.float32), -1, 1))
        a = self._abuf[0]                                   # 액추에이터 지연 적용
        ctrl = self.ctrl_lo + (a + 1.0) * 0.5 * (self.ctrl_hi - self.ctrl_lo)
        b0 = self.data.qpos[self.bq:self.bq + 2].copy()
        hit_now = False
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data); self._pin()
            if self._ball_hit():
                hit_now = True
        mujoco.mj_forward(self.model, self.data)
        d = self.data
        ball = d.qpos[self.bq:self.bq + 2]; ballv = d.qvel[self.bv:self.bv + 2]
        to = self.target - ball; to = to / (np.linalg.norm(to) + 1e-8)
        contact_bonus = 0.5 if (hit_now and not self._bonus_done) else 0.0
        if contact_bonus > 0:
            self._bonus_done = True
        reward = (2.0 * float(np.dot(ballv, to)) + 5.0 * float(np.dot(ball - b0, to))
                  - 0.001 * float(np.sum(a ** 2)) + contact_bonus)
        self._step += 1
        truncated = self._step >= self.max_steps
        disp = ball - self._ball0
        nd = np.linalg.norm(disp)
        aim_err = float(np.degrees(np.arccos(np.clip(np.dot(disp / (nd + 1e-8), self.cmd_dir), -1, 1)))) if nd > 0.05 else 180.0
        info = {"ball_disp": float(nd), "proj": float(np.dot(disp, self.cmd_dir)),
                "aim_err_deg": aim_err, "hit": bool(hit_now or self._bonus_done)}
        return self._obs(), reward, False, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
