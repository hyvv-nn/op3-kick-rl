# env_v6.py  [B(골대 조준) 스캐폴딩 — 2026-06-17 Cowork 사전작성. 미학습 검증은 사용자 터미널]
# ─────────────────────────────────────────────────────────────────────────────
# env_v5(=v5-A) 복제 + 골대 인식/보상. v5-A의 성질(오른다리 액션 마스킹·골반 pin·물리자유) 유지
# → v5-A warm-start(warmstart_v6.py)와 호환. v5-B처럼 관절을 물리고정하지 않음(헤드라인 정책 계승).
#
# ★ 단계적 설계 (계획서 B-2 "한 번에 다 넣지 말 것" 준수). 전부 플래그로 제어:
#   Stage 1 — 골대 인식만:  obs 48→50 (goal_rel 2차원 추가), 보상은 v5-A와 *완전 동일*(w_goal=w_settle=0,
#             goal_as_target=False). 목적: "골대를 봐도 킥 성능 유지" 확인 + warm-start 안착.
#   Stage 2 — 골 보상:      w_goal>0 → 기하 골인(goal_def.is_goal_geometric) 1회 보너스. goal_as_target=True면
#             target을 골대(중심 또는 골문 내 샘플)로 옮겨 기존 reach/align shaping이 골을 향하게 함(sparse 방지).
#             goal_randomize=True면 골문 y범위 내 표적 샘플 → 일반화.
#   Stage 3 — (선택) 안정화/강화: w_vel>0(공속도=파워↑) + w_arate↑(jerk↓) + w_settle>0(settle_after 이후 전체
#             관절 정적; kick_window=0과 함께). "약함/느림/휘적" 3불만 동시 대응.
#             ※ settle은 에피소드를 킥 이후까지 연장(kick_window 크게/None)해야 학습됨. "약한 킥으로 쉽게 정지"
#               보상해킹 위험 → w_settle 작게, goal/reach 우세 유지. goal 수렴 후에만 켤 것.
#
# ★ 불변식: env_v5는 절대 건드리지 않음(v5-A 모델·헤드라인 보호). B 변경은 이 파일에서만.
#   obs 50 = joint40 + ball6 + target_rel2 + goal_rel2.  (env_v5 _obs 주석 [v6/B 슬롯]과 정합)
import os, collections
import numpy as np, mujoco, gymnasium as gym
from gymnasium import spaces
import goal_def
HERE = os.path.dirname(os.path.abspath(__file__))


class OP3KickEnvV6(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 40}

    def __init__(self, scene=None, pin_z=0.33, frame_skip=5, max_steps=300,
                 aim_range=0.4, d_min=1.0, d_max=1.8, kick_window=None,
                 w_reach=5.0, w_align=0.5, w_arate=0.05, w_jvel=5e-4, w_ctrl=1e-3,
                 w_height=0.0, w_squash=0.0, squash_dz=0.03, squash_dxy=0.12,
                 w_goal=0.0, w_vel=0.0, w_settle=0.0, settle_after=20,
                 goal_as_target=False, goal_randomize=False,
                 full_action=False, obs_noise_std=0.0, act_latency=0, render_mode=None):
        scene = scene or os.path.join(HERE, "op3_kick_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(scene); self.data = mujoco.MjData(self.model)
        self.fs, self.max_steps, self.render_mode = frame_skip, max_steps, render_mode
        self._root = np.array([0, 0, pin_z, 1, 0, 0, 0.0])
        self.aim_range, self.d_min, self.d_max = aim_range, d_min, d_max
        self.kick_window = kick_window
        self.w_reach, self.w_align, self.w_arate, self.w_jvel, self.w_ctrl = w_reach, w_align, w_arate, w_jvel, w_ctrl
        self.w_height, self.w_squash, self.squash_dz, self.squash_dxy = w_height, w_squash, squash_dz, squash_dxy
        # ★ v6 신규
        self.w_goal, self.w_vel, self.w_settle, self.settle_after = w_goal, w_vel, w_settle, settle_after
        self.goal_as_target, self.goal_randomize = goal_as_target, goal_randomize
        self.full_action = full_action
        self.obs_noise_std, self.act_latency = obs_noise_std, act_latency
        m = self.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.bq, self.bv = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        self.foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_ank_roll_link")
        self.ball_gid = m.geom("ball_geom").id; self.floor_gid = m.geom("floor").id
        self.lo = m.actuator_ctrlrange[:, 0].copy(); self.hi = m.actuator_ctrlrange[:, 1].copy()
        self.neutral = 0.5 * (self.lo + self.hi)

        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(m.nu)]
        self.act_names = names
        self.kick_idx = [i for i, n in enumerate(names)
                         if n.startswith("r_") and any(k in n for k in ("hip", "knee", "ank"))]
        if len(self.kick_idx) == 0:
            self.kick_idx = list(range(max(0, m.nu - 6), m.nu))
        self.kick_idx = np.array(self.kick_idx, dtype=int)
        # 비킥 관절 qvel dof (settle 보상용) — kick_idx에서 유도, 하드코딩 금지
        kick_set = set(self.kick_idx.tolist())
        self.nonkick_vdof = np.array([6 + i for i in range(m.nu) if i not in kick_set], dtype=int)

        self.act_dim = m.nu if full_action else len(self.kick_idx)
        self.action_space = spaces.Box(-1, 1, (self.act_dim,), np.float32)
        # ★ obs 50 = joint40 + ball6 + target_rel2 + goal_rel2
        self.observation_space = spaces.Box(-np.inf, np.inf, (50,), np.float32)
        self._abuf = collections.deque()

    def _pin(self):
        self.data.qpos[0:7] = self._root; self.data.qvel[0:6] = 0.0

    def _place_goal(self):
        """골대 중심을 설정(단일 출처=goal_def). goal_randomize면 골문 y 범위 내에서 샘플."""
        gp = goal_def.goal_pos().astype(np.float32).copy()
        if self.goal_randomize:
            half = goal_def.GOAL_WIDTH / 2.0 * 0.8          # 가장자리 여유 20%
            gp[1] = goal_def.GOAL_Y_CENTER + np.random.uniform(-half, half)
        self.goal_pos = gp

    def _obs(self):
        d = self.data
        jq, jv = d.qpos[7:27], d.qvel[6:26]
        ball = d.qpos[self.bq:self.bq + 3]; ballv = d.qvel[self.bv:self.bv + 3]
        target_rel = (self.target - ball[:2])
        goal_rel = (self.goal_pos - ball[:2])                # ★ v6 추가(2)
        o = np.concatenate([jq, jv, ball, ballv, target_rel, goal_rel]).astype(np.float32)
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

    def reset_bonus(self): self._bonus = False; self._goaled = False

    def _ctrl_from_action(self, a):
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
        self.ball0 = self.data.qpos[self.bq:self.bq + 2].copy()
        self._place_goal()
        if self.goal_as_target:                              # Stage2: 표적을 골대로
            self.target = self.goal_pos.copy()
            d = self.goal_pos - self.ball0
            self.theta = float(np.arctan2(d[1], d[0])); self.D = float(np.linalg.norm(d))
        else:                                                # Stage1: v5-A와 동일(theta/D 랜덤)
            th = np.random.uniform(-self.aim_range, self.aim_range)
            D = np.random.uniform(self.d_min, self.d_max)
            self.theta, self.D = th, D
            self.target = self.ball0 + D * np.array([np.cos(th), np.sin(th)])
        self._prevdist = float(np.linalg.norm(self.target - self.ball0))
        self.t = 0; self._bonus = False; self._goaled = False
        self._aprev = np.zeros(self.act_dim, np.float32); self._cstep = -1
        self._abuf = collections.deque([np.zeros(self.act_dim, np.float32)] * (self.act_latency + 1),
                                       maxlen=self.act_latency + 1)
        return self._obs(), {"theta": float(self.theta), "D": float(self.D)}

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
        ball_z = float(d.qpos[self.bq + 2]); foot = d.xpos[self.foot]
        dist = float(np.linalg.norm(self.target - ball))
        tdir = self.target - ball; tdir = tdir / (np.linalg.norm(tdir) + 1e-8)
        align = float(np.dot(ballv, tdir)) if np.linalg.norm(ballv) > 0.05 else 0.0
        arate = float(np.sum((a - self._aprev) ** 2))
        jvel = float(np.sum(d.qvel[6:26] ** 2))
        height_err = abs(float(foot[2]) - ball_z)
        dxy = float(np.linalg.norm(foot[:2] - ball))
        squash = 1.0 if (float(foot[2]) > ball_z + self.squash_dz and dxy < self.squash_dxy) else 0.0
        bonus = 0.5 if (hit and not self._bonus) else 0.0
        if bonus: self._bonus = True
        if hit and self._cstep < 0:
            self._cstep = self.t
        # ★ v6 Stage2: 기하 골인 1회 보너스
        scored = goal_def.is_goal_geometric(ball)
        goal_bonus = self.w_goal if (scored and not self._goaled) else 0.0
        if scored: self._goaled = True
        # ★ v6 파워: 공 평면속도 보상(="공이 천천히 간다" 직접 대응). 킥 초반 강타 유도.
        ball_speed = float(np.linalg.norm(ballv))
        vel_term = self.w_vel * ball_speed
        # ★ v6 정착(선택): settle_after 스텝 이후 전체 관절속도 페널티(="차고나서 휘적댐" 대응).
        #   시간기반(_cstep 아님) — 공이 시작부터 발에 닿아 _cstep이 step0에 박히는 함정 회피.
        #   ※ kick_window=0(종료없음)과 함께 써야 학습됨(킥 후 구간이 분포 안에 들어옴).
        settle_pen = 0.0
        if self.w_settle > 0.0 and self.t >= self.settle_after:
            settle_pen = float(np.linalg.norm(d.qvel[6:26]))
        reward = (self.w_reach * (self._prevdist - dist) + self.w_align * align
                  - self.w_arate * arate - self.w_jvel * jvel - self.w_ctrl * float(np.sum(a ** 2))
                  - self.w_height * height_err - self.w_squash * squash + bonus
                  + goal_bonus + vel_term - self.w_settle * settle_pen)
        self._prevdist = dist; self._aprev = a
        terminated = (self.kick_window is not None and self._cstep >= 0 and (self.t - self._cstep) >= self.kick_window)
        self.t += 1
        truncated = self.t >= self.max_steps
        info = {"dist_to_target": dist, "hit": bool(hit or self._bonus),
                "final_err": float(np.linalg.norm(ball - self.target)),
                "foot_z": float(foot[2]), "ball_z": ball_z, "squash": squash,
                "goal": bool(self._goaled), "goal_rel": (self.goal_pos - ball).tolist(),
                "ball_speed": float(np.linalg.norm(ballv)),
                "theta": float(self.theta), "D": float(self.D)}
        return self._obs(), reward, terminated, truncated, info
