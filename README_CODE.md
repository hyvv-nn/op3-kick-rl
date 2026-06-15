# P3 코드 v2 (디벨롭 + 샌드박스 검증완료) — 사용법

> 요청 요약: 엄밀성을 끌어올린 v2 코드. 모든 파일 **2026-06-15 리눅스 샌드박스에서 실제 OP3로 헤드리스 검증**.
> ⚠ 검증 = Linux/CPU. 사용자 = Windows. 로직은 동일, **뷰어·장시간 학습·렌더는 노트북에서**.

## ★ 중요: 지금 도는 5M 학습은 v2로 재시작
v2에서 **env가 바뀌었다**(현실 공 물리, 공 spawn offset 0.15, 목표조건부, 관측노이즈·지연, 보상에 접촉보너스, info 키 `ball_x`→`proj`). 따라서 v1으로 도는 현재 학습은 **중단하고 v2로 재학습**해야 정합한다. 학습이 빠르니(12k step=8.6s, 5M≈수십분~1시간대) 부담 없다.

## 0. v2에서 바뀐 것 (디벨롭 핵심)
| 추가/변경 | 내용 | 검증 |
|-----------|------|------|
| 현실 공 물리 | 질량 0.05→**0.15kg**, 반지름 0.07, 마찰↑ + **spawn offset 0.15**(발 겹침 방지) | 무동작 disp 0.17 vs 차기 0.61 = 비-trivial 확인 |
| 목표조건부(조준) | 매 에피소드 목표 방위각 무작위 + obs의 target_dir로 조건화 | 조준 분석 가능(aim_err) |
| sim2real 현실성 | **관측 노이즈 + 액추에이터 지연**(수준별, Peng2018) | 크래시 없음 확인 |
| anti-해킹 보상 | 결과보상(공속도·전진) + 최초접촉 1회 보너스(접촉횟수 보상 X) | 보상 sanity |
| 해석적 조준 | r_hip_yaw로 조준(부호 실측: 음(-)=+theta) | AIMED 10.3° < NO-AIM 13.3° |
| **analytic_tune.py** | U0 비거리 최대 파라미터 그리드 탐색 → 강한 baseline | BEST proj=0.852, {amp_hip:1.0, amp_knee:0.2, t_swing:0.15} |
| **eval_sim2sim.py** | 학습 분포 밖(OOD U4~U6) 강건성 밴드 | in-dist+OOD 표 생성 |

## 1. 파일 (10개)
`model_inspect.py` · `op3_kick_scene.xml` · `op3_kick_env.py`(v2) · `domain_randomization.py`(v2) · `analytic_kick.py`(v2) · **`analytic_tune.py`** · `train.py`(v2) · `eval.py`(v2) · **`eval_sim2sim.py`** · `README_CODE.md`

## 2. 검증된 모델 사실 (그대로 사용)
- nq=27 nv=26 nu=20. 액추 순서 head(2)→arms(6)→legs(12). kp=21.1, ctrl±π, force±5.
- **전방차기**: r_hip_pitch(idx16) 양(+), r_knee(17) 양(+). **조준**: r_hip_yaw(14) 음(-)=+theta(+y).
- 골반 고정 = 코드 핀(weld는 중력에 실패). 공 spawn = 발 +0.15.

## 3. 배치 + 실행 순서
`P3_code/`의 .py/.xml을 `C:\Users\hyvv_nn\projects\op3_kick\mujoco_menagerie\robotis_op3\`에 복사.
```powershell
cd C:\Users\hyvv_nn\projects\op3_kick ; .venv\Scripts\Activate.ps1
python model_inspect.py                              # (1) 인덱스 대조
python -m mujoco.viewer --mjcf op3_kick_scene.xml    # (2) 시각 확인(발이 공 옆/현실 크기)
python analytic_tune.py                              # (3) 강한 해석적 baseline 파라미터
python train.py --seed 0 --steps 5000000 --n_envs 8 > logs\seed0.log 2>&1   # (4) 직진 차기 학습
python eval.py --model runs\op3_kick_ppo_s0_g0.zip --vecnorm runs\vecnorm_s0_g0.pkl --N 20 --params "{'amp_hip':1.0,'amp_knee':0.2,'t_swing':0.15}"   # (5) 교차점
python eval_sim2sim.py --model runs\op3_kick_ppo_s0_g0.zip --vecnorm runs\vecnorm_s0_g0.pkl --N 10 --params "{'amp_hip':1.0,'amp_knee':0.2,'t_swing':0.15}"   # (6) OOD 강건성
# (L3 조준 과제) python train.py --seed 0 --goal_cond 1 --steps 5000000 ...
```

## 4. ★ 지표 주의 (검증에서 발견한 것)
마찰을 무작위화하면 **'직진 비거리'는 불확실성↑에 오히려 커질 수 있다**(낮은 마찰=더 구름). 따라서 교차점 서사가 가장 깨끗한 지표는 *raw 비거리*가 아니라 **① 목표존 성공률 ② aim_err(목표-상대 정확도) ③ 분산(신뢰성)**이다. eval은 proj+std를, env는 aim_err를 제공하니, *성공률/정확도/분산* 축에서 곡선을 그려라(해석적은 불확실성↑에 분산 급증, RL+DR은 완만 → 신뢰성 교차점).

## 5. 사용자 튜닝 (샌드박스가 못 한 것)
1. 뷰어로 공 크기·위치·발 정렬 최종 확인(현실 RoboCup 공 규격 확인 권장).
2. `analytic_tune.py` 결과를 baseline으로 고정.
3. 학습량 3~5M/시드, 곡선 보며 증감. 톡치기 시 결과보상 비중↑.
4. 성공률/aim_err 기준으로 교차점·sim2sim 그림 작성.

*v2 검증 로그: env(obs48·조준·노이즈·지연·접촉보상) PASS, analytic 조준 PASS, tuner PASS, sim2sim OOD PASS. 직진 비거리 지표의 비단조성 발견 → 성공률/정확도 권장.*
