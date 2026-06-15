# setup.ps1 — P3 op3-kick 환경 배치 (Windows PowerShell)
# 실행: 레포 루트에서  .\setup.ps1   (먼저 venv 활성화 + pip install -r requirements.txt 권장)
$ErrorActionPreference = "Stop"

# 1) MuJoCo Menagerie(OP3만 sparse) clone — 이미 있으면 건너뜀
if (-not (Test-Path "mujoco_menagerie")) {
    Write-Host "[setup] cloning mujoco_menagerie (OP3 sparse)..."
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
    Push-Location mujoco_menagerie
    git sparse-checkout set robotis_op3
    Pop-Location
} else {
    Write-Host "[setup] mujoco_menagerie already present, skip clone."
}

# 2) 코드/씬(src\)을 robotis_op3\ 로 복사 (op3.xml meshdir='assets' 때문에 같은 폴더여야 함)
$dst = "mujoco_menagerie\robotis_op3"
Copy-Item -Path src\*.py, src\*.xml -Destination $dst -Force
Write-Host "[setup] copied src\*.py, src\*.xml -> $dst"

# 3) 로그 폴더
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

Write-Host ""
Write-Host "[setup] done. 다음:"
Write-Host "  cd $dst"
Write-Host "  python model_inspect.py        # 인덱스 확인"
Write-Host "  python analytic_tune.py        # 강한 해석적 baseline"
Write-Host "  python train.py --seed 0 --steps 5000000 --n_envs 8 > ..\..\logs\s0.log 2>&1"
Write-Host "  python eval.py --model runs\op3_kick_ppo_s0_g0.zip --vecnorm runs\vecnorm_s0_g0.pkl --N 20"
