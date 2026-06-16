# goal_def.py  [B 사전장치 — A-eval과 B-보상이 공유할 골대 정의·판정]
# scene XML(op3_kick_scene.xml)의 골대 geom 수치와 일치가 단일 진실.
# ★ v5-A 불변: 이 모듈은 어디에도 자동 연결되지 않음(상수+함수만). env_v5의 obs(48)/보상 변경 없음.
import numpy as np

# ── 골대 기하 (op3_kick_scene.xml 과 동기화) ──
GOAL_LINE_X   = 1.5      # 골라인 x (m)
GOAL_Y_CENTER = -0.07    # 골 중심 y = 킥 라인 (m)
GOAL_WIDTH    = 1.2      # 골폭 (m): y ∈ [center-W/2, center+W/2]
GOAL_HEIGHT   = 0.8      # 시각용 높이 (m)


def goal_pos():
    """골 중심 평면좌표 — B의 obs/보상이 끌어쓰는 단일 출처."""
    return np.array([GOAL_LINE_X, GOAL_Y_CENTER], dtype=np.float32)


def is_goal_geometric(ball_xy, goal_x=GOAL_LINE_X, y_center=GOAL_Y_CENTER, width=GOAL_WIDTH):
    """공이 골라인 통과 & 골폭 안 → 기하학적 골인(거리 의존). B의 보상/일반화 eval용."""
    bx, by = float(ball_xy[0]), float(ball_xy[1])
    return (bx >= goal_x) and (abs(by - y_center) <= width / 2.0)


def is_goal_directional(proj, aim_err_deg, min_reach=0.6, ang_tol_deg=15.0):
    """현재 A-eval과 동일 정의(거리무관 방향). eval_v5a_final.py:94 를 함수화."""
    return (proj >= min_reach) and (aim_err_deg < ang_tol_deg)


if __name__ == "__main__":
    # 자체 점검 — B 사전장치 동작 확인
    print("goal_pos      =", goal_pos())
    print("골폭 y범위    = [%.2f, %.2f]" % (GOAL_Y_CENTER - GOAL_WIDTH / 2,
                                            GOAL_Y_CENTER + GOAL_WIDTH / 2))
    checks = [
        ("골라인 넘고 골폭 안",  is_goal_geometric([1.6, -0.07]), True),
        ("골라인 못 넘음",        is_goal_geometric([1.0, -0.07]), False),
        ("골폭 벗어남(왼쪽)",     is_goal_geometric([1.6,  0.60]), False),
        ("방향 골(정밀·전진)",    is_goal_directional(0.8, 10.0),  True),
        ("방향 빗나감(각도 큼)",  is_goal_directional(0.8, 20.0),  False),
    ]
    ok = all(got == exp for _, got, exp in checks)
    for name, got, exp in checks:
        print("  [%s] %-22s got=%s exp=%s" % ("OK" if got == exp else "X", name, got, exp))
    print("self-test:", "PASS" if ok else "FAIL")
