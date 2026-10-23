import subprocess, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
CWD = r"D:\桌面\工程实践\GNN分子性质预测"

def run(args, timeout=300):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", cwd=CWD, timeout=timeout)

tmp = ["_push2.py", "_verify2.py", "_verify2.txt", "_verify_repo.py", "_verify_repo.txt"]
run(["git", "rm", "--cached", "-q"] + tmp)
run(["git", "add", "-A"])
st = run(["git", "status", "--short"])
print(st.stdout.strip())
r = run(["git", "commit", "-m", "chore: 移除部署过程临时验证脚本"])
print("commit exit:", r.returncode)
if (r.stderr or "").strip():
    print("err:", r.stderr.strip()[-300:])
for i in range(1, 3):
    r = run(["git", "push"])
    print("push exit:", r.returncode)
    if (r.stderr or "").strip():
        print("err:", r.stderr.strip()[-300:])
    if r.returncode == 0:
        print("PUSH_OK")
        break
# 删除本地临时文件
import os
for f in tmp:
    p = os.path.join(CWD, f)
    if os.path.exists(p):
        os.remove(p)
print("local temp files removed")
