# env_v4.py  [미검증 스켈레톤 — 다음 세션이 재학습 후 검증]
# v4 핵심 = "흐느적댐(flailing)" 해결 + 명령 일반화.
#  v3 조준 RL 육안: 다리를 계속 휘젓다 우연히 공 맞힘(aim ±60°=무작위). 원인=보상에 흔들기 처벌·임펄스 종료 없음.
#  → v4 보상: 목표지점 접근 + 방향정렬 + 【흔들기 처벌: action-rate·관절속도 페널티】 + 접촉보너스.
#  → v4 종료: kick_window 설정 시 첫 접촉 후 N스텝에 종료(차기=단발 이벤트). eval은 None(공 정지까지 coast).
#  근거: legged_gym(Rudin 2022)의 action_rate·dof_vel 평활 페널티. 목표=(방향 θ, 거리 D) 일반화, obs에 target_rel(방향+거리).
import os, collections
import numpy as np, mujoco, gymnasium as gym
from gymnasium import spaces
HERE = os.path.dirname(os.path.abspath(__file__))


class OP3KickEnvV4(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 40}

    def __init__(self, scene=None, pin_z=0.33, frame_skip=5, max_steps=300,
                 aim_range=0.4, d_min=1.0, d_max=1.8, kick_window=None,
                 w_reach=5.0, w_align=0.5, w_arate=0.05, w_jvel=5e-4, w_ctrl=1e-3,
                 obs_noise_std=0.0, act_latency=0, render_mode=None):
        scene = scene or os.path.join(HERE, "op3_kick_scene.xml")   # v2 씬(robotis_op3에 배치 시)
        self.model = mujoco.MjModel.from_xml_path(scene); self.data = mujoco.MjData(self.model)
        self.fs, self.max_steps, self.render_mode = frame_skip, max_steps, render_mode
        self._root = np.array([0, 0, pin_z, 1, 0, 0, 0.0])
        self.aim_range, self.d_min, self.d_max = aim_range, d_min, d_max
        self.kick_window = kick_window                              # None=eval coast / 정수=학습 임펄스 종료
        self.w_reach, self.w_align, self.w_arate, self.w_jvel, self.w_ctrl = w_reach, w_align, w_arate, w_jvel, w_ctrl
        self.obs_noise_std, self.act_latency = obs_noise_std, act_latency
        m = self.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.bq, self.bv = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        self.foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_ank_roll_link")
        self.ball_gid = m.geom("ball_geom").id; self.floor_gid = m.geom("floor").id
        self.lo = m.actuator_ctrlrange[:, 0].copy(); self.hi = m.actuator_ctrlrange[:, 1].copy()
        self.action_space = spaces.Box(-1, 1, (m.nu,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (48,), np.float32)  # joint40+ball6+target_rel2
        self._abuf = collections.deque()

    def _pin(self):
        self.data.qpos[0:7] = self._root; self.data.qvel[0:6] = 0.0

    def _obs(self):
        d = self.data
        jq, jv = d.qpos[7:27], d.qvel[6:26]
        ball = d.qpos[self.bq:self.bq + 3]; ballv = d.qvel[self.bv:self.bv + 3]
        target_rel = (self.target - ball[:2])                       # ★실벡터(방향+거리)
        o = np.concatenate([jq, jv, ball, ballv, target_rel]).astype(np.float32)
        if self.obs_noise_std > 0:
            o = o + np.random.normal(0, self.obs_noise_std, o.shape).astype(np.float32)
        return o

    def _hit(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.ball_gid in (c.geom1, c.geom2):
                ot = c.geom2 if c.geom1 == self.ball_gid else c.geom1
                if ot != self.floor_gid:
                    return True
        return False

    def reset_bonus(self): self._bonus = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data); self._pin(); mujoco.mj_forward(self.model, self.data)
        fx, fy, _ = self.data.xpos[self.foot]
        self.data.qpos[self.bq:self.bq + 3] = [fx + 0.15, fy, 0.07]
        self.data.qpos[self.bq + 3:self.bq + 7] = [1, 0, 0, 0]; self.data.qvel[self.bv:self.bv + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        th = np.random.uniform(-self.aim_range, self.aim_range)
        D = np.random.uniform(self.d_min, self.d_max)
        self.theta, self.D = th, D
        self.ball0 = self.data.qpos[self.bq:self.bq + 2].copy()
        self.target = self.ball0 + D * np.array([np.cos(th), np.sin(th)])
        self._prevdist = float(np.linalg.norm(self.target - self.ball0))
        self.t = 0; self._bonus = False; self._aprev = np.zeros(self.model.nu, np.float32); self._cstep = -1
        self._abuf = collections.deque([np.zeros(self.model.nu, np.float32)] * (self.act_latency + 1),
                                       maxlen=self.act_latency + 1)
        return self._obs(), {"theta": float(th), "D": float(D)}

    def step(self, action):
        self._abuf.append(np.clip(np.asarray(action, np.float32), -1, 1)); a = self._abuf[0]
        ctrl = self.lo + (a + 1.0) * 0.5 * (self.hi - self.lo)
        hit = False
        for _ in range(self.fs):
            self.data.ctrl[:] = ctrl; mujoco.mj_step(self.model, self.data); self._pin()
            if self._hit(): hit = True
        mujoco.mj_forward(self.model, self.data)
        d = self.data
        ball = d.qpos[self.bq:self.bq + 2]; ballv = d.qvel[self.bv:self.bv + 2]
        dist = float(np.linalg.norm(self.target - ball))
        tdir = self.target - ball; tdir = tdir / (np.linalg.norm(tdir) + 1e-8)
        align = float(np.dot(ballv, tdir)) if np.linalg.norm(ballv) > 0.05 else 0.0
        arate = float(np.sum((a - self._aprev) ** 2))                 # 흔들기 처벌1: action 변화율
        jvel = float(np.sum(d.qvel[6:26] ** 2))                       # 흔들기 처벌2: 관절속도
        bonus = 0.5 if (hit and not self._bonus) else 0.0
        if bonus: self._bonus = True
        reward = (self.w_reach * (self._prevdist - dist) + self.w_align * align
                  - self.w_arate * arate - self.w_jvel * jvel - self.w_ctrl * float(np.sum(a ** 2)) + bonus)
        self._prevdist = dist; self._aprev = a
        if hit and self._cstep < 0:
            self._cstep = self.t
        terminated = (self.kick_window is not None and self._cstep >= 0 and (self.t - self._cstep) >= self.kick_window)
        self.t += 1
        truncated = self.t >= self.max_steps
        info = {"dist_to_target": dist, "hit": bool(hit or self._bonus),
                "final_err": float(np.linalg.norm(ball - self.target)), "theta": float(self.theta), "D": float(self.D)}
        return self._obs(), reward, terminated, truncated, info
