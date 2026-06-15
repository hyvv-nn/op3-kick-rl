# OP3 Fixed-Base Kick — Analytic vs Learned Control (Crossover & Robustness)

> **One-liner.** On the ROBOTIS OP3 humanoid, we quantitatively compare a physics-based *analytic impulse-maximizing kick* against a *PPO + domain-randomization learned kick*, sweeping environment uncertainty to locate the **crossover** where learning overtakes classical design, and measure **sim-to-sim robustness** beyond the training distribution.
> 로보티즈 OP3 휴머노이드에서 「해석적 임펄스-최대화 차기(고전)」 vs 「PPO+도메인랜덤화 차기(학습)」를 정량 비교하여, 불확실성↑에 학습이 설계를 추월하는 **교차점**과 학습 분포 밖 **sim-to-sim 강건성**을 규명한다.

*포트폴리오 프로젝트 P3 · 박형진(Hyungjin Park), 한양대 기계공학부 · 모든 결과는 시뮬레이션(MuJoCo).*

---

## 배경 (P1 → P3)

선행 단독 소논문 **P1**: 1자유도 관절 서보에서 근궤적 PID(고전) vs PPO(학습)를 정량 비교 → *선형·기지 동역학=고전 우세, 쿨롱·스틱션 비선형=학습 우세*라는 **제어기 선택의 경계**를 발견. 남긴 숙제: ⓐ시뮬만 ⓑPPO 시드 민감성 ⓒ저차원.

**P3**: 그 경계를 *발-공 충돌이 지배하는 고차원 휴머노이드 차기*로 확장·재검증하고, P1의 숙제(시드 재현성·일반화)를 도메인 랜덤화와 다시드로 정면 응답한다. 고정베이스(골반 고정)는 *균형을 범위 밖으로 빼 깨끗한 비교*를 만드는 통제된 설정(P1이 1-DOF를 택한 것과 동일한 정신).

## 연구 질문

- **RQ1**: 환경 불확실성이 커질 때 해석적 vs 학습 차기의 성능은 어떻게 갈리며 *교차점*은 어디인가?
- **RQ2**: 학습 시 DR 범위가 교차점 위치를 얼마나 옮기는가?
- **RQ3**: 도메인 랜덤화가 *시드 분산*(P1 약점)과 *학습 분포 밖(sim-to-sim) 강건성*을 얼마나 개선하는가?
- **RQ4 (스트레치)**: 목표조건부(조준) 차기 / 골반 해제 자유기립으로 경향이 유지되는가?

## 방법

- **환경**: 고정베이스 OP3(20-DoF) 차기 `OP3KickEnv` (MuJoCo, 원시 API). obs(48)=joint(40)+ball(6)+target_dir(2), action(20)=목표관절각.
- **고전(설계)**: 해석적 임펄스-최대화 4-페이즈 차기(원리: Ficht & Behnke 2024) + hip_yaw 조준. `analytic_tune.py`로 U0 비거리 최대 파라미터 자동 탐색 → *강한* 대조군.
- **학습**: PPO (Stable-Baselines3, MlpPolicy [256,256], CPU) + 도메인 랜덤화. 다시드(0,1,2) 보고.
- **불확실성 축 U0~U4**: 공 질량·마찰, 모터 게인, 공 위치 + **액추에이터 지연·관측 노이즈**(sim2real 핵심; Peng 2018 범위). anti-해킹 결과보상(공속도·전진 + 최초접촉 1회).
- **평가**: `eval.py`(in-distribution 교차점), `eval_sim2sim.py`(학습 분포 밖 OOD 강건성). 지표=목표존 성공률·방향오차·분산(직진 비거리는 마찰에 비단조이므로 보조 지표).

## 결과 *(학습 후 그림 삽입 예정)*

- `results/crossover.png` — 불확실성 U0→U4에 따른 해석적 vs 학습 (성공률/정확도). **교차점** 표시.
- `results/sim2sim.png` — in-distribution(U0~U3) vs OOD(U4~U6) 강건성 밴드(평균±표준편차, 다시드).
- `results/dr_ablation.*` — DR on/off가 시드 분산·OOD에 미치는 효과(P1 숙제 응답).
- `results/demo.gif` — 해석적 vs 학습 롤아웃.

## 재현 방법

```powershell
# Python 3.12 권장(3.14 아님). 자세한 사용/튜닝은 README_CODE.md 참조.
py -3.12 -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\setup.ps1                      # mujoco_menagerie(OP3) clone + 코드를 robotis_op3\로 복사
cd mujoco_menagerie\robotis_op3
python model_inspect.py          # 모델 인덱스 확인
python analytic_tune.py          # 강한 해석적 baseline 파라미터
python train.py --seed 0 --steps 5000000 --n_envs 8 > ..\..\logs\s0.log 2>&1
python eval.py --model runs\op3_kick_ppo_s0_g0.zip --vecnorm runs\vecnorm_s0_g0.pkl --N 20 --params "{'amp_hip':1.0,'amp_knee':0.2,'t_swing':0.15}"
python eval_sim2sim.py --model runs\op3_kick_ppo_s0_g0.zip --vecnorm runs\vecnorm_s0_g0.pkl --N 10 --params "{'amp_hip':1.0,'amp_knee':0.2,'t_swing':0.15}"
```

## 레포 구조

```
op3-kick-rl/
  README.md            # (이 문서)
  README_CODE.md       # 상세 사용·튜닝·검증 로그
  GIT_사용법.md         # GitHub 저장 가이드
  requirements.txt
  setup.ps1            # menagerie clone + src\ 코드 배치
  .gitignore           # .venv, mujoco_menagerie, runs, _v1_backup, 모델·영상 제외
  src/                 # v2 코드 .py/.xml (정식 사본, 실행 시 robotis_op3\로 복사됨)
  src_v1/              # v1 코드 .py/.xml (파이프라인 검증용 이전 버전)
  results/             # v2 결과 그림·작은 gif (학습 후 추가)
  results_v1/          # v1 결과 (crossover.png/csv, view_v1.jpg)
```

## 한계 (정직)

모든 결과는 물리 기반 시뮬레이션이며 **실 OP3 하드웨어 검증을 거치지 않았다.** sim-to-sim 강건성은 전이 가능성의 *대리 지표*이지 sim-to-real 보장이 아니다.

## 참고문헌 (핵심)

- Haarnoja et al., *Learning Agile Soccer Skills for a Bipedal Robot with Deep RL*, Science Robotics 2024 (OP3). arXiv:2304.13653
- Ficht & Behnke, *Maximum Impulse Approach to Soccer Kicking for Humanoid Robots*, 2024. arXiv:2412.01480
- Peng et al., *Sim-to-Real Transfer with Dynamics Randomization*, ICRA 2018.
- Raffin et al., *Stable-Baselines3*, JMLR 2021.
- ROBOTIS OP3 — MuJoCo Menagerie (Apache-2.0).
- P1 — 박형진, *외란·파라미터 불확실성 하의 1자유도 관절 서보 제어: 근궤적 PID와 심층강화학습의 정량 비교*, 2026.

## 저자

박형진 (Hyungjin Park), 한양대학교 기계공학부 · ROBOTIS OH! GYM! 지원 포트폴리오(P3).
