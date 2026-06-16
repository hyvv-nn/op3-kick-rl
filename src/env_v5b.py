# env_v5b.py  [미검증 스켈레톤 — viewer로 지지각 실측 후 재학습 검증]
# v5-B = v5-A(액션 마스킹) 실패 대응.
#  v5-A 진단[확정]: 마스킹은 ctrl만 중립일 뿐 관절을 *물리적으로* 안 잠금 → 관성 휘적이 샘. + 지지다리(왼다리) 0rad 직립 → 킥 반작용 못 받쳐 자빠짐.
#  ★처방(방법 C, 검증가능·최소변경): 고정 관절(머리·양팔·왼다리)을 매 스텝 qpos=hold·qvel=0 으로 *강제 고정*(=_pin 확장)
#     → 관절이 관성으로 못 흔들림(물리적 고정과 동등). 왼다리는 *굽힌 지지자세각*(viewer 실측)으로 고정 → 자빠짐 방지.
#     오른다리(actuator 14~19)만 정책 구동. nq/obs 불변(48) → 검증 쉬움.
#  ※ 방법 A(XML weld)는 nq/nv/nu·obs 전면 재작성 필요 → 모델 접근·테스트 불가 환경에선 위험. C가 물리적 고정을 동등 달성.
#  ※ eval(해석적 비교)용: full_action=True → 고정 해제(전신 20관절 구동), 골반만 pin. (학습=마스킹+고정, 평가 해석적=전신)
#  인덱스 근거[확정, 세션11 model_inspect]: 오른다리 actuator 14~19 / qpos qadr 21~26. 고정=actuator 0~13.
import os, collections
import numpy as np, mujoco, gymnasium as gym
from gymnasium import spaces
HERE = os.path.dirname(os.path.abspath(__file__))

# 고정 관절 hold 자세 (rad). 머리·팔=0. ★왼다리=지지자세 [추정 — STEP1 viewer 실측값으로 교체 필수].
HOLD_POSE = {
    "head_pan": 0.0, "head_tilt": 0.0,
    "l_sho_pitch": 0.0, "l_sho_roll": 0.0, "l_el": 0.0,
    "r_sho_pitch": 0.0, "r_sho_roll": 0.0, "r_el": 0.0,
    "l_hip_yaw": 0.0, "l_hip_roll": 0.0, "l_hip_pitch": -0.3,   # ★지지자세(굽힘) [추정]
    "l_knee": 0.6, "l_ank_pitch": -0.3, "l_ank_roll": 0.0,      # ★ viewer로 실측해 교체
}


class OP3KickEnvV5B(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 40}

    def __init__(self, scene=None, pin_z=0.33, frame_skip=5, max_steps=300,
                 aim_range=0.4, d_min=1.0, d_max=1.8, kick_window=None,
                 w_reach=5.0, w_align=0.5, w_arate=0.05, w_jvel=5e-4, w_ctrl=1e-3,
                 w_height=0.0, w_squash=0.0, squash_dz=0.03, squash_dxy=0.12,
                 support_pose=None, full_action=False,
                 obs_noise_std=0.0, act_latency=0, render_mode=None):
        scene = scene or os.path.join(HERE, "op3_kick_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(scene); self.data = mujoco.MjData(self.model)
        self.fs, self.max_steps, self.render_mode = frame_skip, max_steps, render_mode
        self._root = np.array([0, 0, pin_z, 1, 0, 0, 0.0])
        self.aim_range, self.d_min, self.d_max = aim_range, d_min, d_max
        self.kick_window = kick_window
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

        # 정책 구동 = 오른다리 6관절(이름 기반 자동 선별; 세션11 검증=actuator 14~19)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(m.nu)]
        self.act_names = names
        self.kick_idx = [i for i, n in enumerate(names)
                         if n.startswith("r_") and any(k in n for k in ("hip", "knee", "ank"))]
        if len(self.kick_idx) == 0:
            self.kick_idx = list(range(max(0, m.nu - 6), m.nu))
        self.kick_idx = np.array(self.kick_idx, dtype=int)
        kick_set = set(self.kick_idx.tolist())

        # hold 자세 병합(왼다리 지지각 등 사용자 실측값 반영)
        self.hold = dict(HOLD_POSE)
        if support_pose:
            self.hold.update(support_pose)

        # ★ 강제 고정 대상 관절(qpos/qvel 매 스텝 강제) — 이름→qadr/vadr 자동
        self.hold_qadr, self.hold_vadr, self.hold_val = [], [], []
        for jname, val in self.hold.items():
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if j < 0:
                continue                                   # 이름 다르면 skip → STEP3 sanity로 확인
            self.hold_qadr.append(int(m.jnt_qposadr[j])); self.hold_vadr.append(int(m.jnt_dofadr[j]))
            self.hold_val.append(float(val))
        self.hold_qadr = np.array(self.hold_qadr, int); self.hold_vadr = np.array(self.hold_vadr, int)
        self.hold_val = np.array(self.hold_val, np.float64)

        # 고정 actuator(정책 미구동)의 ctrl 목표 = hold 각도(이름에서 _act 제거 후 매칭)
        self.fix_act_idx, self.fix_act_val = [], []
        for i, n in enumerate(names):
            base = n[:-4] if n.endswith("_act") else n
            if i not in kick_set and base in self.hold:
                self.fix_act_idx.append(i); self.fix_act_val.append(float(self.hold[base]))
        self.fix_act_idx = np.array(self.fix_act_idx, int); self.fix_act_val = np.array(self.fix_act_val, np.float64)

        self.act_dim = m.nu if full_action else len(self.kick_idx)
        self.action_space = spaces.Box(-1, 1, (self.act_dim,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (48,), np.float32)  # joint40+ball6+target_rel2
        self._abuf = collections.deque()

    def _pin(self):
        self.data.qpos[0:7] = self._root; self.data.qvel[0:6] = 0.0

    def _hold_fixed(self):
        """골반 pin + 고정 관절(머리·팔·왼다리) qpos=지지각·qvel=0 강제 (=물리적 고정). full_action이면 pin만."""
        self._pin()
        if self.full_action or self.hold_qadr.size == 0:
            return
        self.data.qpos[self.hold_qadr] = self.hold_val
        self.data.qvel[self.hold_vadr] = 0.0

    def _obs(self):
        d = self.data
        jq, jv = d.qpos[7:27], d.qvel[6:26]
        ball = d.qpos[self.bq:self.bq + 3]; ballv = d.qvel[self.bv:self.bv + 3]
        target_rel = (self.target - ball[:2])
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
        if self.full_action:
            return self.lo + (a + 1.0) * 0.5 * (self.hi - self.lo)
        ctrl = np.zeros(self.model.nu)
        if self.fix_act_idx.size:
            ctrl[self.fix_act_idx] = self.fix_act_val          # 고정 actuator → 지지각
        klo, khi = self.lo[self.kick_idx], self.hi[self.kick_idx]
        ctrl[self.kick_idx] = klo + (a + 1.0) * 0.5 * (khi - klo)
        return ctrl

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data); self._pin()
        if not self.full_action and self.hold_qadr.size:       # 시작부터 지지자세로
            self.data.qpos[self.hold_qadr] = self.hold_val
            self.data.qvel[self.hold_vadr] = 0.0
        mujoco.mj_forward(self.model, self.data)
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
            self.data.ctrl[:] = ctrl; mujoco.mj_step(self.model, self.data); self._hold_fixed()
            if self._hit(): hit = True
        mujoco.mj_forward(self.model, self.data)
        d = self.data
        ball = d.qpos[self.bq:self.bq + 2]; ballv = d.qvel[self.bv:self.bv + 2]
        ball_z = float(d.qpos[self.bq + 2]); foot = d.xpos[self.foot]
        dist = float(np.linalg.norm(self.target - ball))
        tdir = self.target - ball; tdir = tdir / (np.linalg.norm(tdir) + 1e-8)
        align = float(np.dot(ballv, tdir)) if np.linalg.norm(ballv) > 0.05 else 0.0
        arate = float(np.sum((a - self._aprev) ** 2))
        jvel = float(np.sum(d.qvel[6:26] ** 2))                # 고정관절은 강제로 0 → 사실상 오른다리만
        height_err = abs(float(foot[2]) - ball_z)
        dxy = float(np.linalg.norm(foot[:2] - ball))
        squash = 1.0 if (float(foot[2]) > ball_z + self.squash_dz and dxy < self.squash_dxy) else 0.0
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
