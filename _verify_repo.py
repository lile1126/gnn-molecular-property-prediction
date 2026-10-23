import subprocess, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def gh(args):
    r = subprocess.run(["gh", "api"] + args, capture_output=True, text=True, encoding="utf-8")
    return r.returncode, (r.stdout or ""), (r.stderr or "")

lines = []
st, out, err = gh(["repos/lile1126/gnn-molecular-property-prediction", "--jq", "{name, full_name, private, default_branch, size, html_url}"])
lines.append("repo: " + out.strip())
st, out, err = gh(["repos/lile1126/gnn-molecular-property-prediction/contents/", "--jq", ".[].name"])
lines.append("root entries:\n" + out.strip())
st, out, err = gh(["repos/lile1126/gnn-molecular-property-prediction/commits?per_page=1", "--jq", ".[0].sha + \" \" + .[0].commit.message.splitlines()[0]"])
lines.append("latest commit: " + out.strip())
open(r"D:\桌面\工程实践\GNN分子性质预测\_verify_repo.txt", "w", encoding="utf-8").write("\n".join(lines))
