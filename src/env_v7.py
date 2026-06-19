# env_v7.py — v7-power: env_v6 + 지수 임팩트 보상(Marew 2024, Eq.38). 순수 RL(잔차 아님).
# ─────────────────────────────────────────────────────────────────────────────
# 원점 설계 근거:
#   - 어젯밤 스윕에서 residual은 약한 analytic baseline 탓에 파워가 안 났다(resb<goalb).
#     -> v7-power는 *순수 RL 최강(goalb)* 에서 warm-start. analytic_kick_v2(잔차용)는 미사용.
#   - 발목 운동사슬 whip은 *보상으로* 유도: 선형 w_vel*|v| 대신 목표방향 성분 공속도에 지수.
#       r_impact = w_impact * ( exp(beta * max(0, v_ball . n)) - 1 ),  cap로 폭주 방지.
#     빠른 임팩트일수록 한계효용 급증 -> 강타 유도(천장 ~2 m/s 안에서 최대화).
#
# ★ 불변식 보존: env_v6를 *상속*만 하고 obs(50)/동역학/기존 보상은 건드리지 않는다.
#   step()에서 super().step() 후 *추가 보너스 항만* 더한다(기존 v6 모델·헤드라인 안전).
#   obs/action 차원이 env_v6와 동일 -> v6goalb/v6clean에서 warm-start 가능.
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from env_v6 import OP3KickEnvV6


class OP3KickEnvV7(OP3KickEnvV6):
    def __init__(self, *args, w_impact=0.5, impact_beta=1.0, impact_cap=8.0, **kw):
        super().__init__(*args, **kw)
        self.w_impact = float(w_impact)        # 지수 임팩트 보상 가중
        self.impact_beta = float(impact_beta)  # 지수 민감도(클수록 강타에 더 보상)
        self.impact_cap = float(impact_cap)    # exp 항 상한(보상 폭주 방지)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)  # v6 물리·보상 그대로
        ball = self.data.qpos[self.bq:self.bq + 2]
        ballv = self.data.qvel[self.bv:self.bv + 2]
        n = self.target - ball
        nn = float(np.linalg.norm(n))
        vdir = float(np.dot(ballv, n / nn)) if nn > 1e-8 else float(np.linalg.norm(ballv))
        imp = max(0.0, vdir)                                  # 목표 방향 성분 공속도(전진만)
        bonus = self.w_impact * (float(np.exp(self.impact_beta * imp)) - 1.0)
        cap = self.w_impact * self.impact_cap
        if bonus > cap:
            bonus = cap
        info["impact_dir"] = vdir
        info["impact_bonus"] = float(bonus)
        return obs, reward + float(bonus), terminated, truncated, info
