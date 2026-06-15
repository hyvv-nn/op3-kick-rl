# analytic_kick.py  v2  [샌드박스 검증완료 2026-06-15: 조준 AIMED 10.3deg < NO-AIM 13.3deg]
# 해석적 임펄스-최대화 차기(Ficht & Behnke 2024 원리) + 조준(hip_yaw).
# 실측 부호: 전방차기 r_hip_pitch(+)/r_knee(+); 조준 r_hip_yaw 음(-)이 +theta(+y)로 튼다.
# 강한 baseline U0 튜닝값(analytic_tune.py 결과): amp_hip=1.0, amp_knee=0.2, t_swing=0.15.
import numpy as np

HIP, KNEE, ANK, YAW = 16, 17, 18, 14


def make_kick(amp_hip=1.0, amp_knee=0.2, t_load=0.15, t_swing=0.15, dt=0.025, aim=0.0, aim_gain=3.0):
    """time t(초)->정규화 목표관절각. 가속 페이즈 quadratic ease-in으로 임팩트 속도 최대화. aim=목표 방위각(rad)."""
    yaw_rad = -aim_gain * aim                          # 조준: -gain*theta (실측 부호)
    def ctrl(t):
        a = np.zeros(20, np.float32)
        if t < t_load:                                # phase 1: load
            k = t / t_load
            a[HIP] = -0.5 * amp_hip * k; a[KNEE] = amp_knee * k
        elif t < t_load + t_swing:                    # phase 2+3: accelerate -> impact
            k = (t - t_load) / t_swing
            a[HIP] = -0.5 * amp_hip + 1.5 * amp_hip * (k * k); a[KNEE] = amp_knee * (1 - k)
        else:                                         # phase 4: recover
            k = min((t - t_load - t_swing) / 0.2, 1.0)
            a[HIP] = amp_hip * (1 - k); a[KNEE] = 0.0
        a[YAW] = yaw_rad
        return np.clip(a / np.pi, -1, 1)
    return ctrl
