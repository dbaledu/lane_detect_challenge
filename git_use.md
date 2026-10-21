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
git clone https://github.com/dbaledu/lane_detect_challenge.git
```

4. 변경된 내용 내려받기
```bash
# 원격 저장소의 최신 커밋을 내려받아 현재 브랜치에 합칩니다
git pull
```


5. 소스 올리기
```bash
# 스테이지 모드로 올리기(이번 커밋에 포함할 파일을 고르는 단계)
git add *

# 커밋(스테이징된 변경사항을 하나의 스냅샷(커밋)으로 내 컴퓨터의 저장소에 기록)
git commit -m "커밋메시지"

# 원격 저장소로 업로드
git push
```

6. 상태확인 등 기타 명령어
```bash
# add 취소
git rm --cached * 

# commit 취소
git rm -r --cached .  

```