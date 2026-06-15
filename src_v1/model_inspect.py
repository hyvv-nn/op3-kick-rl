# model_inspect.py [STEP6 — v1/v2 공통]. mujoco_menagerie/robotis_op3/ 에서 실행.
# 실측(2026-06-15): nq=27 nv=26 nu=20, 액추 head(2)->arms(6)->legs(12), ctrl±π force±5 kp=21.1.
import mujoco
import numpy as np

XML = "op3.xml"
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
print("nq, nv, nu =", m.nq, m.nv, m.nu)
print("--- actuators ---")
for i in range(m.nu):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
print("--- joints (id name type qposadr dofadr) ---")
for i in range(m.njnt):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i),
          int(m.jnt_type[i]), int(m.jnt_qposadr[i]), int(m.jnt_dofadr[i]))
print("ctrlrange[0]=", m.actuator_ctrlrange[0], "forcerange[0]=", m.actuator_forcerange[0],
      "gainprm[0][:3]=", m.actuator_gainprm[0][:3])
