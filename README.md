# OP3 고정베이스 공차기 — 해석적 제어 vs 학습 제어: 다목적 프런티어와 파워 천장

![ROBOTIS OP3 — 학습 킥 (PPO + 도메인 랜덤화)](results_v5a/p3_hero_kick.gif)

> v6(목표 인지 프런티어)와 v7(파워)을 v5-A README에 통합한 판본. v7 파워는 **3시드 확정: 1.86 ± 0.09 m/s**. 보상을 더 키운 스윕에서 기본값이 거의 최적임이 드러났다(공격성을 더 높이면 오히려 파워 저하, §5 참조).

> **한 문장.** ROBOTIS OP3 휴머노이드(골반 고정, MuJoCo)에서 물리 기반 **해석적 임펄스 킥(analytic impulse kick)**과 **PPO + 도메인 랜덤화 학습 킥**을 동일 씬에서 비교한다. 정직한 결론은 **방향 크로스오버(directional crossover)** — 고전은 정밀하게 조준하지만 약하게 차고, 학습 정책은 훨씬 강하게 차며 고전이 닿지 못하는 방향까지 닿는다. 목표 인지(goal-aware) 정책으로 확장하면 **3시드 다목적 프런티어(multi-objective frontier)**(파워·조준·균형이 서로 다른 보상으로 최적화됨)가 나오고, 파워를 **공 launch 속도(ball launch speed)**로 정직하게 재측정하면 학습 킥이 **약 1.86 m/s(3시드) — OP3 planted 하드웨어 천장(~2.0 m/s, DeepMind)의 약 93%**에 도달하며, 남은 격차는 런업(자유 베이스)으로만 좁힐 수 있다.

*포트폴리오 프로젝트 P3 · 박형진, 한양대학교 기계공학부 · 시뮬레이션 전용(MuJoCo); 실물 하드웨어 미검증.*

---

## 데모 영상

전체 해상도 클립은 [`demo/`](demo/)에 있다(상단 히어로 GIF는 파워 챔피언 킥의 트림 영상).

| 클립 | 보여주는 것 |
|------|-------------|
| [`v7_power_kick.mp4`](demo/v7_power_kick.mp4) | v7 — 정직한 공속도 챔피언 (3시드 1.86 m/s, OP3 planted 천장의 약 93%) |
| [`v6goalb_power.mp4`](demo/v6goalb_power.mp4) | v6goalb — 비거리/파워 챔피언 (3시드) |
| [`v6resb_aim.mp4`](demo/v6resb_aim.mp4) | v6resb — 조준 챔피언 (최고 골 성공률) |
| [`classical_kick.mp4`](demo/classical_kick.mp4) | 해석적 임펄스 킥 — 정밀하지만 약함 |
| [`classical_fixed_miss.mp4`](demo/classical_fixed_miss.mp4) | 단일 설계 고전 킥은 측면에서 헛발질(재조준/학습이 필요한 이유) |

---

## 1. 연구 배경 (P2 → P3)

선행 단독 연구 **P2**는 1자유도(1-DoF) 관절 서보에서 근궤적(root-locus) PID(고전)와 PPO(학습)를 비교해 **제어기 선택 경계(controller-selection boundary)**를 밝혔다: 선형·기지 동역학에서는 고전이, 쿨롱·스틱션 비선형에서는 학습이 우세하다. **P3**는 그 경계를 **발–지면 접촉이 지배하는 고차원 휴머노이드 킥**으로 밀어붙인다. 골반을 고정해 전신 균형을 의도적으로 범위 밖으로 두어 해석적 vs 학습 비교를 깨끗하게 유지한다.

## 2. 과제와 방법

- **환경**: 고정베이스 OP3 킥(MuJoCo 3.9, raw API), 매 스텝 env 코드가 골반을 고정. 관측은 v6에서 목표의 상대 위치(`goal_rel`)를 포함해 48→50으로 늘어난다.
- **고전(설계)**: 해석적 임펄스 최대화 킥(Ficht & Behnke), *fixed*(단일 설계)와 *oracle*(목표마다 재조준·재출력).
- **학습**: PPO(SB3, MLP [256,256], CPU) + 도메인 랜덤화(공 질량·마찰, 모터 게인, 액추에이터 지연, 관측 노이즈; Peng).
- **공정 비교**: 동일 씬·동일 공 위치. RL은 **액션 마스킹(action masking)**으로 오른다리(6-DoF)만 구동, 고전은 20-DoF 전체 사용.
- **두 평가 렌즈**: (a) 위치 기반 방향 지표(비거리 reach, 조준 오차, 거리 독립 골)로 크로스오버; (b) **공 launch/peak 속도(m/s)**로 정직한 파워(§5).

## 3. 진행: v1 → v7

| 단계 | 내용 | 결과 |
|------|------|------|
| **v1–v4** | 파이프라인, 현실 물리, 평가축 전환, 매끄러움 페널티 | 보상 해킹 진단·수정; 파워 확보, 휘적임 잔존 |
| **v5-A ★** | **액션 마스킹**(오른다리만, 액션 20→6) | **깨끗 + 강력.** 상체 휘적임을 구조적으로 차단 |
| v5-B / v5.5 (기각) | 물리 제약 / 강한 조준 보상 | 뻣뻣·약함 / 이득 없음 |
| **v6** | **목표 인지**(obs 48→50); v6clean 3시드 floor; Stage-2·residual 변형 | **크로스오버 정량화 + 다목적 프런티어**(§4) |
| **v7** | **파워**: 순수 RL 정책에 지수 임팩트 보상(Marew) | **정직한 공속도 1.72 → 1.86 m/s(3시드), planted 천장의 약 93%**(§5) |

## 4. v6 — 3시드 다목적 프런티어

목표 인지 학습은 하나의 챔피언을 만들지 않는다. **보상마다 다른 축이 최적화**되며, 그 분리는 3시드(mean ± std)에서 확인된다.

| config | 골 성공률 | 비거리 reach (m) | 조준 (°) | 역할 |
|--------|-----------|------------------|----------|------|
| **v6clean** | 0.41 ± 0.05 | 6.39 ± 0.18 | 19.7 ± 1.5 | 균형 floor(견고한 매끄러움, 파워 유지) |
| **v6goalb** (Stage-2) | 0.37 ± 0.05 | **6.81 ± 0.05** | 21.6 ± 0.7 | **파워 챔피언(확정)** |
| **v6resb** (residual 0.5) | **0.49 ± 0.08** | 6.13 ± 0.10 | 20.1 ± 1.3 | **조준 챔피언**(평균적으로 유지) |

**정직한 시드 보정.** v6clean의 단일 시드는 파워 증가처럼 *보였으나*(reach 6.63 m), 3시드 평균(6.39 m)은 그것이 **파워는 유지, 매끄러움은 견고하게 회복**임을 드러냈다 — 겉보기 이득은 시드 운(seed luck)이었다. 헤드라인 수치를 mean ± std로 보고하는 이유가 정확히 이것이다.

**residual 스케일 스윕(우리 가설을 우리가 반박).** residual RL 이론은 10–20% 권한을 권장한다(Johannink). 그러나 우리 셋업에서는 *반대*가 성립했다: 스케일 **0.15는 파워 붕괴**(reach 4.48 m, goal 0.33) — 우리의 해석적 baseline이 약해서(~1 m) residual이 파워를 떠받쳐야 하기 때문. **0.5가 최적**, 0.3이 차선. 문헌 기반 가설을 우리 영역에 맞게 정직하게 교정한 설명 가능한 발견이다.

## 5. v7 — 물리 천장 대비 정직하게 측정한 파워

**reach는 부풀려진 지표였다.** ~6–7 m "reach"는 굴림 산물(거의 무마찰 바닥 + 가벼운 0.15 kg 공)로, 1.6 m/s 킥도 7 m를 굴러간다. 정직한 파워 지표는 **공 launch/peak 속도(m/s)**다.

| 모델 | 공 peak 속도(전체) | 천장 대비 |
|------|--------------------|-----------|
| v6goalb_s2 (파워 챔피언) | 1.72 m/s | 2.0의 86% |
| **v7-power (3시드)** | **1.86 ± 0.09 m/s** | **약 93% of 2.0** |
| v7-power, FIFA 0.43 kg 공 | 1.67 m/s (1시드) | 실제 질량, 약 14% 느림 |

![OP3 planted 천장 대비 정직한 파워](results_v5a/p3_power_ceiling.png)

**메커니즘과 정직한 범위.** v7은 선형 속도 보상을 지수 임팩트 항 `w·(exp(β·v·n) − 1)`(Marew, Eq. 38)로 대체해 순수 RL 정책(파워 챔피언에서 warm-start)에 적용한다. 초선형 보상이 더 강하고 발목까지 동원된 임팩트를 유도한다: 3시드 **1.86 ± 0.09 m/s, baseline 1.72 대비 +8%**. 최고 단일 시드는 1.95 m/s(선호 각도에선 2.0 초과)에 닿았으나 한 시드는 거의 개선이 없어(1.73) 3시드 평균을 정직한 헤드라인으로 둔다 — 시드 규율 재확인. **보상에는 sweet spot이 있다:** 더 강한 변형은 오히려 파워를 *저하*시켰다(w_impact 0.8 / β 1.5 → 1.53 m/s; 1.0 / β 2.0 → 1.40 m/s, 모두 baseline 이하). 따라서 기본값(0.5, β 1.0)이 거의 최적이며 공격성을 더 높이면 역효과다. 실제 질량 FIFA 공(0.43 kg)에서도 v7은 1.67 m/s에 닿아, 파워가 단지 가벼운 공의 산물이 아님을 보인다.

**천장은 물리적 한계이지 튜닝 실패가 아니다.** *동일한* ROBOTIS OP3에서 DeepMind는 planted 킥 천장 **~2.0 m/s**(학습이 스크립트의 3% 이내 = 하드웨어 한계)와 **런업(자유 베이스) 시에만 2.8–3.4 m/s**를 측정했다. 우리는 planted ~1.95 m/s다. 일부 각도에서 2.0을 약간 넘는 값은 우리 sim이 실물보다 미세하게 관대하기 때문(액추에이터 ±5 N·m vs 실물 XM430 stall 4.1 N·m; 가벼운 공)이며 벤치마크를 **이긴 것이 아니다**. **토크는 절대 올리지 않았다** — 그것은 sim-to-real 격차만 키운다. ~2.0 너머의 진짜 프런티어는 런업/자유 베이스 = 향후 과제다.

## 6. 관절 수준 안정성 (3시드)

*어디로·얼마나 멀리*를 넘어 *얼마나 깨끗하게* 킥을 수행하는가. 킥 구간에서 낮을수록 안정적: 비킥 관절 운동, settle 시간, 명령 jerk(모터 부하 대리 지표).

| 제어기 | 비킥 운동 ↓ | settle ↓ | 명령 jerk ↓ | reach ↑ |
|--------|-------------|----------|-------------|---------|
| 고전(해석적) | 4.98 | 0.70 s | **3.06** | 1.06 m |
| RL v5-A (헤드라인) | 31.6 | 60 | 97 | 7.11 m |
| **RL v6clean (3시드)** | **18.5 ± 0.7** | **53 ± 1** | **76 ± 9** | 6.39 |

**마스킹은 제 역할을 하고, v6는 매끄러움을 견고하게 회복한다.** 차는 다리만 구동하면 몸통이 조용해지고, v6clean은 비킥 운동(31.6→18.5)과 jerk(97→76)를 시드 전반에서 더 줄이면서 파워를 유지한다. 강력하고 활동적(학습) vs 정밀하고 부드러움(고전)은 θ-크로스오버 뒤의 동일한 상보성이다.

## 7. 핵심 발견

1. **"덜 제약하는 편이 나았다"**(v5): 마스킹이 탐색 공간을 구조적으로 줄여 보상 기반 억제를 이겼다; 과도한 제약(v5-B)은 킥을 약하게 만들었다.
2. **단일 챔피언은 없다 — 다목적 프런티어**(v6): 파워=goalb, 조준=resb, 균형=clean, 3시드에서 확인.
3. **파워는 물리 천장에 근접**(v7): 정직한 공속도 **1.86 ± 0.09 m/s(3시드) ≈ 93%**(DeepMind 검증 OP3 planted 천장 대비); 기술(지수 임팩트 보상, sweet spot — 더 강하면 저하)로 비물리적 토크 없이 도달.

## 8. 엔지니어링 정직성

- **옳은 것을 측정한다.** reach(굴림 거리)는 파워를 과대평가 → 파워 헤드라인을 **공 launch 속도**로 전환하고 동일 로봇 벤치마크와 비교.
- **3시드 후에만 주장한다.** 단일 시드가 시드 운을 파워 이득으로 오인 → mean ± std로 교정.
- **평가기를 먼저 검증한다.** 초기 두 버그(하드코딩된 뷰어 모델; free-joint qvel 인덱싱 함정)는 지표를 시각 롤아웃과 교차 확인해야만 잡혔다.

## 9. 한계와 향후 과제 (정직)

- **시뮬레이션 전용**; 실물 OP3 미검증. 고정 베이스 = 전신 균형 미측정.
- RL 조준은 좌측(−θ) 약점이 있고, v7 파워 이득은 방향 비대칭.
- **고정된 공 위치.** 공이 매 에피소드 같은 자리로 리셋되므로 현재 정책은 *옮겨진* 공을 안정적으로 차지 못한다(분포 외, OOD). **공 위치 도메인 랜덤화**를 더하면 학습 정책이 옮겨진 공에 적응할 수 있다 — 개방 루프 고전 킥이 재설계 없이 다룰 수 없는 상황이며, 학습 이점의 가장 깔끔한 시연이자 계획된 확장이다.
- **진짜 파워 프런티어는 런업/자유 베이스**(2.8–3.4 m/s, DeepMind)로, 보행과 균형이 더해진 사실상 새 프로젝트다.

## 10. 재현

```
# Python 3.12 (.venv). cd mujoco_menagerie\robotis_op3
$py = "C:\Users\hyvv_nn\projects\op3_kick\.venv\Scripts\python.exe"
# v6 방향 크로스오버 / 프런티어
& $py eval_v6_dir.py --model runs\op3_kick_v6clean.zip --vecnorm runs\vecnorm_v6clean.pkl --N 20 --tag v6clean
# 정직한 파워(공 launch 속도, m/s)
& $py eval_power.py --model runs\op3_kick_v6goalb_s2.zip --vecnorm runs\vecnorm_v6goalb_s2.pkl --goal_cond --N 10 --tag v6goalb_s2
& $py eval_power.py --model runs\op3_kick_v7pow.zip      --vecnorm runs\vecnorm_v7pow.pkl      --goal_cond --N 10 --tag v7pow
# 관찰(파워 vs 조준 vs 고전)
& $py watch_v6.py     --model runs\op3_kick_v6goalb_s2.zip --vecnorm runs\vecnorm_v6goalb_s2.pkl --goal_randomize
& $py watch_v6res.py  --model runs\op3_kick_v6resb_s2.zip  --vecnorm runs\vecnorm_v6resb_s2.pkl  --res_scale 0.5
& $py watch_analytic.py
```

## 11. 참고문헌 (핵심)

- Haarnoja et al., *Learning Agile Soccer Skills for a Bipedal Robot with Deep RL*, Science Robotics. arXiv:2304.13653 — 동일 OP3; planted 2.0 / 런업 2.8–3.4 m/s.
- Marew et al., *A Biomechanics-Inspired Approach to Soccer Kicking for Humanoid Robots*. arXiv:2407.14612 — 운동사슬 휩(whip), 지수 임팩트 보상(Eq. 38).
- Ficht & Behnke, *Maximum Impulse Approach to Soccer Kicking for Humanoid Robots*. arXiv:2412.01480.
- Johannink et al., *Residual Reinforcement Learning for Robot Control*, ICRA. arXiv:1812.03201.
- Peng et al., *Sim-to-Real Transfer with Dynamics Randomization*, ICRA.
- Raffin et al., *Stable-Baselines3*, JMLR. · ROBOTIS OP3 — MuJoCo Menagerie (Apache-2.0).
- P2 — 박형진, *근궤적 PID vs 심층강화학습: 1자유도 관절 서보 제어 비교*.

## 12. 저자

**박형진, 전민제** — 한양대학교 기계공학부 · ROBOTIS OH! GYM! 지원 포트폴리오(P3). (P1·P2·P3 공저.)
