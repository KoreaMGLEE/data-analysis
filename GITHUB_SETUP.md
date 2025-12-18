# GitHub 업로드 가이드

## 1. GitHub에서 저장소 생성

1. https://github.com/new 접속
2. Repository name 입력 (예: `true_diff`)
3. Public 또는 Private 선택
4. **"Initialize this repository with a README"는 체크하지 마세요** (이미 로컬에 파일이 있음)
5. "Create repository" 클릭

## 2. 로컬 저장소를 GitHub에 연결

GitHub에서 저장소를 만든 후 나타나는 URL을 사용하세요:

```bash
# HTTPS 방식 (권장)
git remote add origin https://github.com/YOUR_USERNAME/true_diff.git

# 또는 SSH 방식 (SSH 키가 설정되어 있는 경우)
git remote add origin git@github.com:YOUR_USERNAME/true_diff.git
```

## 3. 코드 업로드

```bash
git branch -M main
git push -u origin main
```

## 완료!

이제 GitHub에서 코드를 확인할 수 있습니다.

---

## 참고: Git 사용자 정보 설정 (선택사항)

처음 Git을 사용하시는 경우, 사용자 정보를 설정하는 것을 권장합니다:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

