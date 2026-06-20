# OP3 Fixed-Base Kick — Analytic vs Learned Control: a Multi-Objective Frontier and the Power Ceiling

![ROBOTIS OP3 — learned kick (PPO + domain randomization)](results_v5a/p3_hero_kick.gif)

> **DRAFT (2026-06-19)** integrating v6 (goal-aware frontier) and v7 (power) into the v5-A README.
> v7 power is now **3-seed confirmed: 1.86 ± 0.09 m/s**. A stronger-reward sweep showed the default is near-optimal (more aggression degrades power; see §5).

> **One-liner.** On the ROBOTIS OP3 humanoid (fixed pelvis, MuJoCo), a physics-based *analytic impulse kick* and a *PPO + domain-randomization learned kick* are compared on the same scene. The honest finding is a **directional crossover** — classical aims precisely but kicks weakly, the learned policy kicks far harder and reaches directions classical cannot. Extending to a goal-aware policy yields a **3-seed multi-objective frontier** (power / aim / balance optimized by different rewards), and an honest re-measurement of power as **ball launch speed** shows the learned kick reaching **~1.86 m/s (3-seed) — about 93% of the OP3 planted hardware ceiling (~2.0 m/s, DeepMind 2024)**, with the remaining gap closable only by run-up (free base).

*Portfolio project P3 · Hyungjin Park, Dept. of Mechanical Engineering, Hanyang University · Simulation only (MuJoCo); no physical hardware tested.*

## Demo videos

Full-resolution clips are in [`demo/`](demo/) (the hero GIF above is a trimmed view of the power-champion kick):

| clip | what it shows |
|------|---------------|
| [`v7_power_kick.mp4`](demo/v7_power_kick.mp4) | v7 — honest peak-speed champion (3-seed 1.86 m/s, ~93% of the OP3 planted ceiling) |
| [`v6goalb_power.mp4`](demo/v6goalb_power.mp4) | v6goalb — reach / power champion (3-seed) |
| [`v6resb_aim.mp4`](demo/v6resb_aim.mp4) | v6resb — aim champion (highest goal rate) |
| [`classical_kick.mp4`](demo/classical_kick.mp4) | analytic impulse kick — precise but weak |
| [`classical_fixed_miss.mp4`](demo/classical_fixed_miss.mp4) | single-design classical misses off-axis (why a re-aimed / learned kick is needed) |

---

## 1. Background (P2 → P3)

A prior solo paper **P2** compared root-locus PID (classical) against PPO (learned) on a 1-DoF joint servo and found a **controller-selection boundary**: classical wins on linear/known dynamics, learning wins under Coulomb/stiction nonlinearity. **P3** pushes that boundary into a *high-dimensional humanoid kick where foot–ball contact dominates*. The fixed pelvis deliberately moves balance out of scope to keep the analytic-vs-learned comparison clean.

## 2. Task & Method

- **Environment**: fixed-base OP3 kick (MuJoCo 3.9, raw API), pelvis pinned by env code each step. Observation grows 48→50 in v6 to include the goal's relative position (`goal_rel`).
- **Classical (design)**: analytic impulse-maximizing kick (Ficht & Behnke 2024), *fixed* (single design) and *oracle* (re-aimed/re-powered per target).
- **Learned**: PPO (SB3, MLP [256,256], CPU) + domain randomization (ball mass/friction, motor gain, actuator latency, observation noise; Peng 2018).
- **Fair comparison**: same scene, same ball position. RL drives only the right leg via **action masking** (6-DoF); classical uses all 20 DoF.
- **Two evaluation lenses**: (a) *position-based directional* metrics (reach, aim error, distance-independent goal) for the crossover; (b) *ball launch/peak speed* (m/s) for honest power (§5).

## 3. Progression: v1 → v7

| Stage | What it is | Outcome |
|------|------------|---------|
| **v1–v4** | Pipeline, realistic physics, eval-axis switch, smoothness penalties | Reward-hacking diagnosed and fixed; power gained, thrash remained |
| **v5-A ★** | **Action masking** (right leg only, action 20→6) | **Clean + strong.** Upper-body flailing blocked *structurally* |
| v5-B / v5.5 (rejected) | Physical constraints / stronger aim reward | Stiff+weak / no gain |
| **v6** | **Goal-aware** (obs 48→50); v6clean 3-seed floor; Stage-2 & residual variants | **Quantified crossover + multi-objective frontier** (§4) |
| **v7** | **Power**: exponential impact reward (Marew 2024) on the pure-RL policy | **Honest ball speed 1.72 → 1.86 m/s (3-seed), ~93% of the planted ceiling** (§5) |

## 4. v6 — a 3-seed multi-objective frontier

Goal-aware training does not produce one champion; **different rewards optimize different axes**, and the split is confirmed across 3 seeds (mean ± std):

| config | goal rate | reach (m) | aim (°) | role |
|---|---|---|---|---|
| **v6clean** | 0.41 ± 0.05 | 6.39 ± 0.18 | 19.7 ± 1.5 | balanced floor (robust smoothness, power held) |
| **v6goalb** (Stage-2) | 0.37 ± 0.05 | **6.81 ± 0.05** | 21.6 ± 0.7 | **power champion (confirmed)** |
| **v6resb** (residual 0.5) | **0.49 ± 0.08** | 6.13 ± 0.10 | 20.1 ± 1.3 | **aim champion** (holds on average) |

**Honest seed correction.** A single seed of v6clean *looked* like a power increase (reach 6.63 m); the 3-seed mean (6.39 m) revealed it as **power held, smoothness robustly recovered** — the apparent gain was seed luck. Headline numbers are reported as mean ± std for exactly this reason.

**Residual-scale sweep (refutes our own guess).** Residual RL theory suggests 10–20% authority (Johannink 2019). Empirically, on our setup the *opposite* held: scale **0.15 collapsed power** (reach 4.48 m, goal 0.33) because our analytic baseline is weak (~1 m), so the residual must carry the power. **0.5 was best**, 0.3 second. An honest, explainable finding that corrects a literature-based hypothesis for our regime.

## 5. v7 — power, measured honestly against the physical ceiling

**reach was an inflated metric.** The ~6–7 m "reach" is a rolling artifact (near-frictionless floor + a light 0.15 kg ball); a 1.6 m/s kick still rolls 7 m. The honest power metric is **ball launch/peak speed (m/s)**.

| model | peak ball speed (overall) | vs ceiling |
|---|---|---|
| v6goalb_s2 (power champion) | 1.72 m/s | 86% of 2.0 |
| **v7-power (3-seed)** | **1.86 ± 0.09 m/s** | **~93% of 2.0** |
| v7-power, FIFA 0.43 kg ball | 1.67 m/s (1 seed) | real-mass, ~14% slower |

![Honest power vs the OP3 planted ceiling](results_v5a/p3_power_ceiling.png)

**Mechanism & honest scope.** v7 replaces the linear speed reward with an exponential impact term `w·(exp(β·v·n) − 1)` (Marew 2024, Eq. 38) on the pure-RL policy (warm-started from the power champion). The superlinear reward drives a harder, ankle-recruited impact: 3-seed **1.86 ± 0.09 m/s, +8% over the 1.72 baseline**. The single best seed reached 1.95 m/s (and >2.0 at favored angles), but one seed barely improved (1.73) — so the 3-seed mean is the honest headline, re-confirming the seed discipline. **The reward has a sweet spot:** stronger variants *degraded* power (w_impact 0.8 / β 1.5 → 1.53 m/s; 1.0 / β 2.0 → 1.40 m/s, both below baseline), so the default (0.5, β 1.0) is near-optimal — more aggression backfires. With a real-mass FIFA ball (0.43 kg) v7 still reaches 1.67 m/s, so the power is not merely a light-ball artifact.

**The ceiling is physical, not a tuning failure.** On the *same* ROBOTIS OP3, DeepMind (Science Robotics 2024) measured a planted-kick ceiling of **~2.0 m/s** (learned within 3% of scripted = hardware limit) and **2.8–3.4 m/s only with a run-up (free base)**. We are at ~1.95 m/s planted. Values slightly above 2.0 at some angles reflect our sim being marginally more permissive than real hardware (actuator ±5 N·m vs real XM430 stall 4.1 N·m; light ball), **not** beating the benchmark. **Torque was never raised** — that would only widen the sim-to-real gap. The genuine frontier beyond ~2.0 is run-up/free-base = future work.

## 6. Joint-level stability (3-seed)

Beyond *where* and *how far*, *how cleanly* the kick executes. Over the kick window, lower = steadier: non-kick joint motion, settle time, command jerk (a motor-load proxy).

| controller | non-kick motion ↓ | settle ↓ | command jerk ↓ | reach ↑ |
|---|---:|---:|---:|---:|
| Classical (analytic) | 4.98 | 0.70 s | **3.06** | 1.06 m |
| RL v5-A (headline) | 31.6 | 60 | 97 | 7.11 m |
| **RL v6clean (3-seed)** | **18.5 ± 0.7** | **53 ± 1** | **76 ± 9** | 6.39 |

**The masking does its job, and v6 recovers smoothness robustly.** Driving only the kicking leg keeps the body quiet; v6clean further cuts non-kick motion (31.6→18.5) and jerk (97→76) across seeds while holding power. Powerful-and-energetic (learned) vs precise-and-gentle (classical) is the same complementarity behind the θ-crossover.

## 7. Key findings

1. **"Less constraining was better"** (v5): masking structurally shrank the search space and beat reward-based suppression; over-constraining (v5-B) made the kick weak.
2. **No single champion — a multi-objective frontier** (v6): power = goalb, aim = resb, balance = clean, confirmed across 3 seeds.
3. **Power is near the physical ceiling** (v7): honest ball speed **1.86 ± 0.09 m/s (3-seed) ≈ 93%** of the DeepMind-verified OP3 planted ceiling; technique (exponential impact reward, at its sweet spot — stronger reward degrades) reached it without non-physical torque.

## 8. Engineering honesty

- **Measure the right thing.** reach (rolling distance) overstates power; we switched the power headline to **ball launch speed** and compared against a same-robot benchmark.
- **Claim only at 3 seeds.** A single seed mislabeled seed-luck as a power gain; corrected by mean ± std.
- **Verify the evaluator before trusting it.** Two earlier bugs (a hard-coded viewer model; a free-joint qvel indexing trap) were caught only by cross-checking metrics against visual rollouts.

## 9. Limitations & future work (honest)

- **Simulation only**; no physical OP3 tested. Fixed base = whole-body balance not measured.
- RL aim has a left-side (−θ) weakness; v7's power gain is direction-asymmetric.
- **Fixed ball placement.** The ball is reset to the same spot each episode, so the current policy would not reliably kick a *moved* ball (out-of-distribution). Adding **ball-position domain randomization** would let the learned policy adapt to a moved ball — a situation the open-loop classical kick cannot handle without redesign. This is the cleanest demonstration of the learning advantage and a planned extension.
- **The real power frontier is run-up / free-base** (2.8–3.4 m/s, DeepMind), which adds locomotion and balance — effectively a new project.

## 10. Reproduce

```powershell
# Python 3.12 (.venv). cd mujoco_menagerie\robotis_op3
$py = "C:\Users\hyvv_nn\projects\op3_kick\.venv\Scripts\python.exe"
# v6 directional crossover / frontier
& $py eval_v6_dir.py --model runs\op3_kick_v6clean.zip --vecnorm runs\vecnorm_v6clean.pkl --N 20 --tag v6clean
# honest power (ball launch speed, m/s)
& $py eval_power.py --model runs\op3_kick_v6goalb_s2.zip --vecnorm runs\vecnorm_v6goalb_s2.pkl --goal_cond --N 10 --tag v6goalb_s2
& $py eval_power.py --model runs\op3_kick_v7pow.zip      --vecnorm runs\vecnorm_v7pow.pkl      --goal_cond --N 10 --tag v7pow
# watch (power vs aim vs classical)
& $py watch_v6.py     --model runs\op3_kick_v6goalb_s2.zip --vecnorm runs\vecnorm_v6goalb_s2.pkl --goal_randomize
& $py watch_v6res.py  --model runs\op3_kick_v6resb_s2.zip  --vecnorm runs\vecnorm_v6resb_s2.pkl  --res_scale 0.5
& $py watch_analytic.py
```

## 11. References (core)

- Haarnoja et al., *Learning Agile Soccer Skills for a Bipedal Robot with Deep RL*, Science Robotics 2024. arXiv:2304.13653 — same OP3; planted 2.0 / run-up 2.8–3.4 m/s.
- Marew et al., *A Biomechanics-Inspired Approach to Soccer Kicking for Humanoid Robots*, 2024. arXiv:2407.14612 — kinematic-chain whip, exponential impact reward (Eq. 38).
- Ficht & Behnke, *Maximum Impulse Approach to Soccer Kicking for Humanoid Robots*, 2024. arXiv:2412.01480.
- Johannink et al., *Residual Reinforcement Learning for Robot Control*, ICRA 2019. arXiv:1812.03201.
- Peng et al., *Sim-to-Real Transfer with Dynamics Randomization*, ICRA 2018.
- Raffin et al., *Stable-Baselines3*, JMLR 2021. · ROBOTIS OP3 — MuJoCo Menagerie (Apache-2.0).
- P2 — Hyungjin Park, *Root-Locus PID vs Deep RL for 1-DoF Joint Servo Control*, 2026.

## 12. Authors

**Hyungjin Park, Minje Jeon** — Dept. of Mechanical Engineering, Hanyang University · ROBOTIS OH! GYM! application portfolio (P3). (Co-authors on P1, P2, and P3.)
