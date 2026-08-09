"""Drive the probing submits + score reading through the Kaggle CLI.

Requires ~/.kaggle/kaggle.json (the user's own API token) and the `kaggle`
package installed in the venv. Matches scores back to probes by fileName.

  python 21_kaggle_submit.py submit [--only base|probes] [--sleep 3]
  python 21_kaggle_submit.py scores        # pull submissions, fill log.json
Then run `20_lb_probe.py solve` / `craft`.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "exp" / "probes"
LOG = PROBE_DIR / "log.json"
BASE_CSV = ROOT / "exp" / "base_autogluon_0.2531.csv"
COMP = "labo-iii-2026-ba"
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")


def _run(args):
    return subprocess.run([KAGGLE, *args], capture_output=True, text=True)


def _already_scored():
    """fileNames that already have a public score on Kaggle (idempotent submit)."""
    r = _run(["competitions", "submissions", "-c", COMP, "--csv", "--page-size", "200"])
    import csv
    import io
    done = set()
    if r.returncode == 0 and r.stdout.strip():
        for row in csv.DictReader(io.StringIO(r.stdout)):
            fn = row.get("fileName")
            ps = row.get("publicScore")
            if fn and ps:
                done.add(fn)
    return done


def submit(which="all", sleep=3.0):
    log = json.loads(LOG.read_text())
    jobs = []
    if which in ("all", "base"):
        jobs.append((BASE_CSV, "base_autogluon_0.2531"))
    if which in ("all", "probes"):
        jobs += [(PROBE_DIR / p["file"], p["file"][:-4]) for p in log["probes"]]
    done = _already_scored()
    skipped = [p for p, _ in jobs if p.name in done]
    jobs = [(p, m) for p, m in jobs if p.name not in done]
    if skipped:
        print(f"skipping {len(skipped)} already-scored files")
    print(f"submitting {len(jobs)} files to {COMP} (sleep {sleep}s between)")
    for i, (path, msg) in enumerate(jobs, 1):
        r = _run(["competitions", "submit", "-c", COMP, "-f", str(path), "-m", msg])
        out = (r.stdout + r.stderr).strip().replace("\n", " ")
        ok = "successfully submitted" in out.lower()
        print(f"[{i}/{len(jobs)}] {path.name}: {'OK' if ok else 'CHECK -> ' + out[:160]}")
        if not ok and ("429" in out or "limit" in out.lower()):
            print("  ! daily submission limit hit — stop and resume later")
            break
        time.sleep(sleep)


def scores():
    """Pull the submissions table and fill scores into log.json by fileName."""
    r = _run(["competitions", "submissions", "-c", COMP, "--csv", "--page-size", "200"])
    if r.returncode != 0 or not r.stdout.strip():
        print("ERROR reading submissions:", (r.stdout + r.stderr)[:300])
        return
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(r.stdout)))
    # newest first; keep the most recent score per fileName
    def col(row, *names):
        for n in names:
            for k in row:
                if k.lower().replace(" ", "") == n:
                    return row[k]
        return None
    by_name = {}
    for row in rows:
        fn = col(row, "filename", "file")
        ps = col(row, "publicscore")
        if fn and ps and fn not in by_name:
            try:
                by_name[fn] = float(ps)
            except ValueError:
                pass
    log = json.loads(LOG.read_text())
    got = 0
    b = by_name.get("base_autogluon_0.2531.csv")
    if b is not None:
        log["base_score"] = b
        got += 1
    for p in log["probes"]:
        s = by_name.get(p["file"])
        if s is not None:
            p["score"] = s
            got += 1
    LOG.write_text(json.dumps(log, indent=2))
    missing = [p["file"] for p in log["probes"] if p["score"] is None]
    print(f"filled {got} scores. base_score={log['base_score']}")
    if log["base_score"] is not None:
        tag = "OK (== 0.255, identidad confirmada)" if abs(log["base_score"] - 0.255) < 0.001 else \
              f"!! esperaba 0.255, base NO es la 0.2531"
        print(f"  base identity: {tag}")
    if missing:
        print(f"missing ({len(missing)}): {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "submit":
        which = "all"
        sleep = 3.0
        if "--only" in sys.argv:
            which = sys.argv[sys.argv.index("--only") + 1]
        if "--sleep" in sys.argv:
            sleep = float(sys.argv[sys.argv.index("--sleep") + 1])
        submit(which, sleep)
    elif cmd == "scores":
        scores()
