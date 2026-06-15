# GitHub 저장 가이드 (P3)

> 요청 요약: 작업을 계속 GitHub에 저장해도 되는지 / 방법 / 내가 대신 해줄 수 있는지.

## 0. 결론
- **그렇다, 계속 저장하라.** 포폴이 어차피 *공개 GitHub 링크*를 요구하므로 일찍 레포를 만들어 마일스톤마다 커밋하는 게 이득(버전관리 + 제출물 자동 준비).
- **내가 네 GitHub에 직접 push는 못 한다.** 이유: ① push=*공개 게시*라 네 계정 로그인/권한이 필요(나는 네 자격증명 입력·대리 게시 불가) ② 학습 코드·결과가 *네 PC*(`C:\Users\hyvv_nn\projects\op3_kick`)에 있고 내 작업 환경에서 접근 불가.
- **대신 내가 한 것**: 레포 구조·`.gitignore`·아래 명령/커밋 흐름을 다 준비. 너는 사실상 *복붙*만 하면 된다.
- **GitHub 커넥터 정정**: 「GitHub 연동」 커넥터는 *존재하고 연결돼 있다*. 단 이건 **Chat(레포 파일 첨부)·Projects(레포를 컨텍스트로 동기화)·Claude Code(레포 선택·브랜치·PR 추적)**용이다 — *이 Cowork 세션엔 GitHub 도구가 붙지 않아 여기서 내가 push는 못 한다*(확인함). **이 레포 코딩+git 루프는 Claude Code로 하는 게 정석**(네 프로젝트 폴더에서 실제 git 실행 가능).

## 1. 권장 레포 구조 (대용량은 빼고 코드+결과그림만)
```
op3-kick-rl/
  README.md            # P1→P3 배경·RQ·교차점·sim2sim·재현법·OH GYM 후크
  requirements.txt     # mujoco, gymnasium, stable-baselines3, imageio, tensorboard ...
  setup.ps1            # mujoco_menagerie clone + 코드 복사 (아래)
  src/                 # P3_code의 .py/.xml (정식 사본)
  results/             # crossover.png, sim2sim.png, gif 등 (작은 것만)
  .gitignore           # 이미 만들어 둠(.venv, mujoco_menagerie, runs, *.zip ...)
```
> ⚠ 코드는 실행 시 `mujoco_menagerie/robotis_op3/`에 있어야 하므로, **정식 사본은 `src/`에 두고 `setup.ps1`이 menagerie를 clone한 뒤 복사**하게 한다. menagerie(수백 MB)는 커밋하지 않는다.

## 2. 최초 1회 (레포 만들고 올리기)
```powershell
# (A) GitHub에서 빈 레포 'op3-kick-rl' 생성 (웹) — 또는 gh CLI 설치 시: gh repo create op3-kick-rl --public
cd C:\Users\hyvv_nn\projects\op3_kick
git init
# .gitignore를 프로젝트 루트에 복사(P3_code\.gitignore 사용)
git add .
git commit -m "init: OP3 fixed-base kick — analytic vs RL crossover (P3)"
git branch -M main
git remote add origin https://github.com/<너의아이디>/op3-kick-rl.git
git push -u origin main      # 최초 push 시 브라우저/PAT로 로그인(네가 1회 인증)
```
> 인증: 처음 push할 때 GitHub 로그인 창(또는 Personal Access Token)이 뜬다. **이건 네가 직접** 해야 한다(비밀번호/토큰은 내가 다루지 않음).

## 3. 계속 저장 (마일스톤마다 — 이 3줄 반복)
```powershell
git add -A
git commit -m "feat: 교차점 곡선 1차 (성공률 축, seed0)"   # 메시지는 그날 한 일
git push
```
권장 커밋 시점: 환경 안착 / 교차점 1차 / 다시드+DR ablation / sim2sim / 조준 / README·공개. (자잘한 변경마다 말고 *의미 단위*로.)

## 4. 큰 파일 주의
- 모델(`*.zip`,`*.pkl`)·로그·영상(`*.mp4`)·menagerie는 `.gitignore`로 제외(레포 비대·푸시 실패 방지).
- 보여줄 결과는 **그림(png)·작은 gif만** `results/`에 커밋(`git add -f results\demo.gif`).
- 정말 모델/영상을 올리려면 GitHub *Release 자산* 또는 git-lfs 사용.

## 5. 내가 더 도울 수 있는 것
- `README.md`·`requirements.txt`·`setup.ps1` 초안 작성(요청 시 바로).
- 커밋 메시지·레포 설명문·레포 구조 정리.
- (단, 실제 `git push`·공개는 네 PC에서 네 계정으로 — 이건 위임 불가.)

*요약: 저장은 적극 권장, 방법은 §2~3 복붙, push 인증만 네 몫. 나머지 준비는 끝나 있다.*
