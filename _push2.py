import subprocess, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
CWD = r"D:\桌面\工程实践\GNN分子性质预测"

def run(args, timeout=300):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", cwd=CWD, timeout=timeout)

# 从索引移除内部文件
run(["git", "rm", "--cached", "-q",
     "reports/_collect_report_data.json",
     "reports/_page_script.js",
     "reports/_page_script2.js"])
run(["git", "add", "-A"])
st = run(["git", "status", "--short"])
print(st.stdout.strip())
r = run(["git", "commit", "-m", "chore: 排除内部数据收集与页面调试脚本"])
print("commit exit:", r.returncode)
if r.stderr.strip():
    print("err:", r.stderr.strip()[-400:])

for i in range(1, 4):
    print(f"=== push attempt {i} ===", flush=True)
    r = run(["git", "push"])
    print("exit", r.returncode, flush=True)
    if (r.stdout or "").strip():
        print("out:", r.stdout.strip()[-400:], flush=True)
    if (r.stderr or "").strip():
        print("err:", r.stderr.strip()[-400:], flush=True)
    if r.returncode == 0:
        print("PUSH_OK", flush=True)
        break
