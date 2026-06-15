# domain_randomization.py  v2  [샌드박스 검증완료 2026-06-15]
# 불확실성 축(U0~U4) = 동역학(질량/마찰/게인/공위치) + sim2real(관측노이즈/액추에이터지연).
# 범위 출처: Peng 2018(ICRA) + MuJoCo Playground G1 + legged_gym. 질량 반폭은 공 0.15kg 기준 상향.
import numpy as np
import gymnasium as gym

LEVELS = {
    0: dict(m=0.00, f=0.00, x=0.00, g=0.00, noise=0.000, lat=0),   # U0: 불확실성 없음(해석적 튜닝 기준)
    1: dict(m=0.02, f=0.10, x=0.01, g=0.05, noise=0.005, lat=0),
    2: dict(m=0.04, f=0.20, x=0.02, g=0.10, noise=0.010, lat=1),
    3: dict(m=0.06, f=0.30, x=0.03, g=0.15, noise=0.015, lat=1),
    4: dict(m=0.08, f=0.40, x=0.04, g=0.20, noise=0.020, lat=2),
}


class DomainRandomize(gym.Wrapper):
    """에피소드마다 공 질량/마찰, 바닥 마찰, 모터 게인, 공 x오프셋 무작위화.
    관측노이즈/액추에이터지연은 수준별로 env에 주입(매 스텝 적용)."""

    def __init__(self, env, level=0):
        super().__init__(env)
        self.level = level
        self.L = LEVELS[level]
        u = self.unwrapped
        u.obs_noise_std = self.L["noise"]
        u.act_latency = self.L["lat"]
        m = u.model
        self._bg = m.geom("ball_geom").id
        self._bb = m.body("ball").id
        self._g0 = m.actuator_gainprm[:, 0].copy()
        self._f0 = float(m.geom_friction[self._bg, 0])
        self._m0 = float(m.body_mass[self._bb])
        self._fl0 = float(m.geom("floor").friction[0])

    def reset(self, **kw):
        m = self.unwrapped.model
        L = self.L
        U = np.random.uniform
        m.body_mass[self._bb] = max(0.03, self._m0 + U(-L["m"], L["m"]))
        m.geom_friction[self._bg, 0] = max(0.05, self._f0 + U(-L["f"], L["f"]))
        m.geom("floor").friction[0] = max(0.05, self._fl0 + U(-L["f"], L["f"]))
        m.actuator_gainprm[:, 0] = self._g0 * (1.0 + U(-L["g"], L["g"], size=self._g0.shape))
        self.unwrapped._dx = U(-L["x"], L["x"])
        return self.env.reset(**kw)
