#### git 사용법
document : https://docs.github.com/ko/get-started/git-basics/setting-your-username-in-git

0. 토큰 발급

1. git사용자 등록
```bash
git config --global user.name "Mona Lisa"
git config --global user.email "@ .."
```
3. 리포지토리 내려받기
```bash
git remote add origin https://github.com/dbaledu/lane_detect_challenge.git
git branch -M main
```

5. 소스 올리기
```
git add *
git commit -m "커밋메시지"
git push
```
