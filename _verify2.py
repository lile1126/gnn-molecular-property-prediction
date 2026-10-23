import subprocess, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def gh(args):
    r = subprocess.run(["gh", "api"] + args, capture_output=True, text=True, encoding="utf-8")
    return r.returncode, (r.stdout or ""), (r.stderr or "")

lines = []
st, out, err = gh(["repos/lile1126/gnn-molecular-property-prediction/commits", "--jq", ".[0] | {sha: .sha[0:9], msg: .commit.message.splitlines()[0], files: [.files[].filename]}"])
lines.append("commit: " + out.strip())
if err.strip():
    lines.append("err: " + err.strip()[:300])
st, out, err = gh(["repos/lile1126/gnn-molecular-property-prediction/contents/src", "--jq", ".[].name"])
lines.append("src entries: " + out.strip().replace("\n", ", "))
st, out, err = gh(["repos/lile1126/gnn-molecular-property-prediction/contents/reports", "--jq", ".[].name"])
lines.append("reports entries: " + out.strip().replace("\n", ", "))
open(r"D:\桌面\工程实践\GNN分子性质预测\_verify2.txt", "w", encoding="utf-8").write("\n".join(lines))
