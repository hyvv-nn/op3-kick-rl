# domain_randomization.py  v1 [초기 버전. 정식=상위 폴더 v2(노이즈·지연 포함)]
import numpy as np
import gymnasium as gym

LEVELS = {
    0: dict(m=0.00, f=0.00, x=0.00, g=0.00),
    1: dict(m=0.010, f=0.10, x=0.010, g=0.05),
    2: dict(m=0.020, f=0.20, x=0.020, g=0.10),
    3: dict(m=0.030, f=0.30, x=0.030, g=0.15),
    4: dict(m=0.040, f=0.40, x=0.040, g=0.20),
}


class DomainRandomize(gym.Wrapper):
    def __init__(self, env, level=0):
        super().__init__(env)
        self.level = level
        self.L = LEVELS[level]
        m = self.unwrapped.model
        self._bgid = m.geom("ball_geom").id
        self._bbid = m.body("ball").id
        self._g0 = m.actuator_gainprm[:, 0].copy()
        self._fric0 = float(m.geom_friction[self._bgid, 0])
        self._mass0 = float(m.body_mass[self._bbid])
        self._floor0 = float(m.geom("floor").friction[0])

    def reset(self, **kw):
        m = self.unwrapped.model
        L = self.L
        U = np.random.uniform
        m.body_mass[self._bbid] = max(0.01, self._mass0 + U(-L["m"], L["m"]))
        m.geom_friction[self._bgid, 0] = max(0.05, self._fric0 + U(-L["f"], L["f"]))
        m.geom("floor").friction[0] = max(0.05, self._floor0 + U(-L["f"], L["f"]))
        m.actuator_gainprm[:, 0] = self._g0 * (1.0 + U(-L["g"], L["g"], size=self._g0.shape))
        self.unwrapped._dx = U(-L["x"], L["x"])
        return self.env.reset(**kw)
