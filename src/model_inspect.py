# model_inspect.py  [STEP6 — 샌드박스 실측 완료. 이 출력으로 인덱스를 재확인하라]
# 실행: 이 파일을 mujoco_menagerie/robotis_op3/ 에 두고  python model_inspect.py
# 실측 결과(2026-06-15, op3.xml 단독): nq=27 nv=26 nu=20.
#   액추 순서: 0 head_pan,1 head_tilt, 2 l_sho_pitch,3 l_sho_roll,4 l_el, 5 r_sho_pitch,6 r_sho_roll,7 r_el,
#             8 l_hip_yaw,9 l_hip_roll,10 l_hip_pitch,11 l_knee,12 l_ank_pitch,13 l_ank_roll,
#             14 r_hip_yaw,15 r_hip_roll,16 r_hip_pitch,17 r_knee,18 r_ank_pitch,19 r_ank_roll.
#   ctrlrange ±π, forcerange ±5, gainprm[0]=kp=21.1.  joint qpos=qpos[7:27], qvel=qvel[6:26].
import mujoco
import numpy as np

XML = "op3.xml"   # 또는 "op3_kick_scene.xml"
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
print("nq, nv, nu =", m.nq, m.nv, m.nu)
print("--- actuators (id : name) ---")
for i in range(m.nu):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
print("--- joints (id name type qposadr dofadr) ---")
for i in range(m.njnt):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i),
          "type=", int(m.jnt_type[i]), "qadr=", int(m.jnt_qposadr[i]), "vadr=", int(m.jnt_dofadr[i]))
print("ctrlrange[0]=", m.actuator_ctrlrange[0], "forcerange[0]=", m.actuator_forcerange[0],
      "gainprm[0][:3]=", m.actuator_gainprm[0][:3])
