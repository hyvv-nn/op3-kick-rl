# env_v6res.py — Residual 하이브리드 env (moonshot): analytic baseline + 학습 잔차
# ─────────────────────────────────────────────────────────────────────────────
# 명제 완결("best of both"): 고전의 정밀 조준 + 학습의 파워를 한 정책에.
#   baseline = analytic make_kick(aim=theta)(t)  (20관절, yaw=-3*theta로 조준 prior)
#   policy   = 우다리 kick_idx(6관절)에 *잔차만* 출력
#   ctrl     = clip(baseline + res_scale*residual, -1,1) → full 20관절 매핑
#
# ★ 불변식 보존: env_v6를 *상속*만 하고 obs(50)·보상·동역학은 건드리지 않는다.
#   바뀌는 것은 _ctrl_from_action(액션→ctrl 경로) 하나뿐. obs/action 차원이 v6clean과
#   동일(6-dim)하므로 v6clean에서 warm-start 가능.
#
# 가설(정직): 잔차가 analytic 조준 prior를 물려받아 *동일 파워에서 aim↓*. 단 analytic 파워가
#   약해(reach~1m) 잔차가 파워 대부분을 떠안을 수 있어 효과 불확실 → moonshot.
#   res_scale로 잔차 권한 조절(1.0=완전권한, <1=보정 위주).
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from env_v6 import OP3KickEnvV6
from analytic_kick import make_kick


class OP3KickEnvV6Res(OP3KickEnvV6):
    def __init__(self, *args, res_scale=1.0, amp_hip=1.0, amp_knee=0.2, t_swing=0.15, **kw):
        kw["full_action"] = False                      # 6-dim 우다리 잔차(=v6clean과 동일 act 차원)
        super().__init__(*args, **kw)
        self.res_scale = float(res_scale)
        self._akw = dict(amp_hip=amp_hip, amp_knee=amp_knee, t_swing=t_swing)
        self._dt = self.fs * float(self.model.opt.timestep)   # env step당 실시간(초) = frame_skip*timestep
        self._analytic = make_kick(aim=0.0, **self._akw)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # 표적 방위(theta)로 analytic baseline 조준(상위 reset이 self.theta 설정 후 호출)
        self._analytic = make_kick(aim=float(self.theta), **self._akw)
        return obs, info

    def _ctrl_from_action(self, a):
        t_sec = self.t * self._dt
        base = np.asarray(self._analytic(t_sec), np.float32)          # 20-dim, [-1,1]
        comb = base.copy()
        comb[self.kick_idx] = np.clip(
            base[self.kick_idx] + self.res_scale * np.asarray(a, np.float32), -1.0, 1.0)
        return self.lo + (comb + 1.0) * 0.5 * (self.hi - self.lo)     # full 20관절 매핑
