# env_v5.py  [미검증 스켈레톤 — 재학습으로 검증]
# v5 = v4 flailing 미해결(상체 휘적 + 공 깔림 + 일부 미접촉) 대응 재설계.
#  ★핵심1 (전민제 안, 1순위): 차는 다리(오른다리)만 정책이 구동, 나머지(머리·팔·왼다리)는 중립 ctrl 고정
#     → action 차원 20→6 축소. 상체 휘적을 *구조적으로* 차단 + 탐색 난이도↓(학습 안정).
#  ★핵심2 (2순위, 보상): 발-공 높이정렬 + 공중심 타격 보상, 발이 공 위로 가면 squash 페널티 → "깔기"→"차기".
#     기본 w_height=w_squash=0 (=v5-A: 액션마스킹만). >0이면 v5-B(보상 추가). "변경 최소화" 위해 A부터.
#  안정화: train_v5에서 learning_rate↓(1e-4)·ent_coef>0(탐색 유지).
#  obs는 v4와 동일 (48,) 유지(joint40+ball6+target_rel2) → 의미 동일, 단 action 차원이 달라 모델 재사용 불가(재학습 필수).
#  eval(고전 비교)용: full_action=True 로 생성하면 마스킹 해제(20-dim) → 해석적 킥(20관절)을 같은 씬에서 구동.
import os, collections
import numpy as np, mujoco, gymnasium as gym
from gymnasium import spaces
HERE = os.path.dirname(os.path.abspath(__file__))


class OP3KickEnvV5(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 40}

    def __init__(self, scene=None, pin_z=0.33, frame_skip=5, max_steps=300,
                 aim_range=0.4, d_min=1.0, d_max=1.8, kick_window=None,
                 w_reach=5.0, w_align=0.5, w_arate=0.05, w_jvel=5e-4, w_ctrl=1e-3,
                 w_height=0.0, w_squash=0.0, squash_dz=0.03, squash_dxy=0.12,
                 full_action=False, obs_noise_std=0.0, act_latency=0, render_mode=None):
        scene = scene or os.path.join(HERE, "op3_kick_scene.xml")   # v2 씬(robotis_op3에 배치 시)
        self.model = mujoco.MjModel.from_xml_path(scene); self.data = mujoco.MjData(self.model)
        self.fs, self.max_steps, self.render_mode = frame_skip, max_steps, render_mode
        self._root = np.array([0, 0, pin_z, 1, 0, 0, 0.0])
        self.aim_range, self.d_min, self.d_max = aim_range, d_min, d_max
        self.kick_window = kick_window                              # None=eval coast / 정수=학습 임펄스 종료
        self.w_reach, self.w_align, self.w_arate, self.w_jvel, self.w_ctrl = w_reach, w_align, w_arate, w_jvel, w_ctrl
        self.w_height, self.w_squash, self.squash_dz, self.squash_dxy = w_height, w_squash, squash_dz, squash_dxy
        self.full_action = full_action
        self.obs_noise_std, self.act_latency = obs_noise_std, act_latency
        m = self.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.bq, self.bv = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        self.foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_ank_roll_link")
        self.ball_gid = m.geom("ball_geom").id; self.floor_gid = m.geom("floor").id
        self.lo = m.actuator_ctrlrange[:, 0].copy(); self.hi = m.actuator_ctrlrange[:, 1].copy()
        self.neutral = 0.5 * (self.lo + self.hi)                    # 고정 관절 중립 ctrl(중점) ※튜닝 가능

        # ★ 차는 다리(오른다리) 액추에이터 인덱스 — 이름 기반 자동 탐색(추측 금지: 모델에서 직접 읽음)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(m.nu)]
        self.act_names = names
        self.kick_idx = [i for i, n in enumerate(names)
                         if n.startswith("r_") and any(k in n for k in ("hip", "knee", "ank"))]
        if len(self.kick_idx) == 0:                                 # fallback: legs(12) 중 뒤쪽 6 = 오른다리
            self.kick_idx = list(range(max(0, m.nu - 6), m.nu))
        self.kick_idx = np.array(self.kick_idx, dtype=int)

        self.act_dim = m.nu if full_action else len(self.kick_idx)
        self.action_space = spaces.Box(-1, 1, (self.act_dim,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (48,), np.float32)  # joint40+ball6+target_rel2
        self._abuf = collections.deque()

    def _pin(self):
        self.data.qpos[0:7] = self._root; self.data.qvel[0:6] = 0.0

    def _obs(self):
        d = self.data
        jq, jv = d.qpos[7:27], d.qvel[6:26]
        ball = d.qpos[self.bq:self.bq + 3]; ballv = d.qvel[self.bv:self.bv + 3]
        target_rel = (self.target - ball[:2])
        # [v6/B 슬롯] goal_rel = goal_def.goal_pos() - ball[:2]
        #   → o = concat([..., target_rel, goal_rel]) 로 obs 48→50. env_v5(v5-A)에선 추가 금지(모델 로드 불가).
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

    def _ctrl_from_action(self, a):
        """정책 action → 전체 actuator ctrl. masked 모드면 차는 다리만, 나머지는 중립."""
        if self.full_action:
            return self.lo + (a + 1.0) * 0.5 * (self.hi - self.lo)
        ctrl = self.neutral.copy()
        klo, khi = self.lo[self.kick_idx], self.hi[self.kick_idx]
        ctrl[self.kick_idx] = klo + (a + 1.0) * 0.5 * (khi - klo)
        return ctrl

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
        # [v6/B 훅] self.goal_pos = goal_def.goal_pos(); self.goal_width = goal_def.GOAL_WIDTH
        #   (A에선 미사용. 별도 _place_goal()로 빼면 B에서 골대 위치 랜덤화 쉬움.)
        self._prevdist = float(np.linalg.norm(self.target - self.ball0))
        self.t = 0; self._bonus = False
        self._aprev = np.zeros(self.act_dim, np.float32); self._cstep = -1
        self._abuf = collections.deque([np.zeros(self.act_dim, np.float32)] * (self.act_latency + 1),
                                       maxlen=self.act_latency + 1)
        return self._obs(), {"theta": float(th), "D": float(D)}

    def step(self, action):
        self._abuf.append(np.clip(np.asarray(action, np.float32), -1, 1)); a = self._abuf[0]
        ctrl = self._ctrl_from_action(a)
        hit = False
        for _ in range(self.fs):
            self.data.ctrl[:] = ctrl; mujoco.mj_step(self.model, self.data); self._pin()
            if self._hit(): hit = True
        mujoco.mj_forward(self.model, self.data)
        d = self.data
        ball = d.qpos[self.bq:self.bq + 2]; ballv = d.qvel[self.bv:self.bv + 2]
        ball_z = float(d.qpos[self.bq + 2])
        foot = d.xpos[self.foot]                                    # 발 위치(3D)
        dist = float(np.linalg.norm(self.target - ball))
        tdir = self.target - ball; tdir = tdir / (np.linalg.norm(tdir) + 1e-8)
        align = float(np.dot(ballv, tdir)) if np.linalg.norm(ballv) > 0.05 else 0.0
        arate = float(np.sum((a - self._aprev) ** 2))               # 흔들기 처벌1: action 변화율
        jvel = float(np.sum(d.qvel[6:26] ** 2))                     # 흔들기 처벌2: 관절속도(고정관절은 ≈0)
        # ★ v5 추가 보상항 (기본 가중치 0 → 켜면 적용)
        height_err = abs(float(foot[2]) - ball_z)                   # 발-공 높이차(중심타격 유도)
        dxy = float(np.linalg.norm(foot[:2] - ball))                # 발-공 수평거리
        squash = 1.0 if (float(foot[2]) > ball_z + self.squash_dz and dxy < self.squash_dxy) else 0.0  # 위에서 누름
        bonus = 0.5 if (hit and not self._bonus) else 0.0
        if bonus: self._bonus = True
        reward = (self.w_reach * (self._prevdist - dist) + self.w_align * align
                  - self.w_arate * arate - self.w_jvel * jvel - self.w_ctrl * float(np.sum(a ** 2))
                  - self.w_height * height_err - self.w_squash * squash + bonus)
        self._prevdist = dist; self._aprev = a
        if hit and self._cstep < 0:
            self._cstep = self.t
        terminated = (self.kick_window is not None and self._cstep >= 0 and (self.t - self._cstep) >= self.kick_window)
        self.t += 1
        truncated = self.t >= self.max_steps
        info = {"dist_to_target": dist, "hit": bool(hit or self._bonus),
                "final_err": float(np.linalg.norm(ball - self.target)),
                "foot_z": float(foot[2]), "ball_z": ball_z, "squash": squash,
                "theta": float(self.theta), "D": float(self.D)}
        return self._obs(), reward, terminated, truncated, info
