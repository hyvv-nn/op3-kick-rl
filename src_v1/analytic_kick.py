# analytic_kick.py  v1 [초기 버전 — 직진만. 정식=상위 폴더 v2(조준 포함)]
# 해석적 임펄스-최대화 4-페이즈 차기(Ficht & Behnke 2024 원리). 오른다리 r_hip_pitch(16)/r_knee(17).
import numpy as np

HIP, KNEE, ANK = 16, 17, 18


def make_kick(amp_hip=1.0, amp_knee=0.5, t_load=0.15, t_swing=0.18, dt=0.025):
    def ctrl(t):
        a = np.zeros(20, np.float32)
        if t < t_load:                                # load
            k = t / t_load
            a[HIP] = -0.5 * amp_hip * k; a[KNEE] = amp_knee * k
        elif t < t_load + t_swing:                    # accelerate -> impact
            k = (t - t_load) / t_swing
            a[HIP] = -0.5 * amp_hip + 1.5 * amp_hip * (k * k); a[KNEE] = amp_knee * (1 - k)
        else:                                         # recover
            k = min((t - t_load - t_swing) / 0.2, 1.0)
            a[HIP] = amp_hip * (1 - k); a[KNEE] = 0.0
        return np.clip(a / np.pi, -1, 1)
    return ctrl
