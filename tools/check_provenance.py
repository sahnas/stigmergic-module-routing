"""Provenance checks, run from the repository root: python tools/check_provenance.py

1. Code hashes: every 'file (sha256 xxxx)' quoted in preregistrations/*.md is recomputed, first in original-fr/
   (files as executed for experiments 2 to 7), then in src/ and src/jepa/ (experiments 8 to 13).
2. Seed ledger: which seeds appear in which result files, and every overlap between test seeds of different
   experiments (development and tuning files are listed separately).
"""
import glob, hashlib, json, os, re, collections, subprocess

def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]

print('1. Code hashes quoted in preregistrations')
bad = 0
historical = 0
for pre in sorted(glob.glob('preregistrations/*.md')):
    for name, h in re.findall(r'([\w/-]+\.py) \(sha256 ([0-9a-f]{16})\)', open(pre).read()):
        base = os.path.basename(name)
        found = [p for p in ['original-fr/' + base, 'src/' + base, 'src/jepa/' + base] + glob.glob('kaggle/*/' + base) if os.path.exists(p) and sha16(p) == h]
        recovered = None
        # A later executable may legitimately differ from the preregistered
        # snapshot. Recover the exact quoted bytes from this branch's history,
        # report the difference explicitly, and never replace the current file.
        if not found and os.path.isfile(name):
            commits = subprocess.check_output(['git', 'log', '--format=%H', '--', name], text=True).splitlines()
            for commit in commits:
                blob = subprocess.run(['git', 'show', f'{commit}:{name}'], capture_output=True)
                if blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest()[:16] == h:
                    recovered = f'{commit[:12]}:{name}'
                    break
        if found:
            state = f'OK current ({found[0]})'
        elif recovered:
            historical += 1
            state = f'OK historical ({recovered}); current differs ({sha16(name)})'
        else:
            bad += 1
            state = 'MISMATCH: quoted version not recovered'
        print(f'  {os.path.basename(pre):28s} {base:24s} {h}  {state}')
print(f'  -> {bad} unrecovered quoted hash(es); {historical} recovered only in Git history\n')
print('  Recovery verifies source availability, not which version an unlogged run executed.\n')

print('2. Seed ledger')
ledger = collections.defaultdict(set)
for f in sorted(glob.glob('results/**/*.jsonl', recursive=True)):
    for line in open(f):
        r = json.loads(line)
        if 'seed' in r:
            ledger[f].add(r['seed'])
dev = lambda f: any(k in f for k in ('dev', 'tuning', 'lr_check'))
for f, s in ledger.items():
    s = sorted(s)
    print(f'  {"dev " if dev(f) else "test"}  {f:62s} {len(s):3d} seeds  {s[0]}..{s[-1]}')
print('\n  Overlaps between test-seed files of different experiments:')
tests = {f: s for f, s in ledger.items() if not dev(f)}
exp = lambda f: f.split('/')[1]
found = False
for a in tests:
    for b in tests:
        if a < b and exp(a) != exp(b) and tests[a] & tests[b]:
            found = True
            ov = sorted(tests[a] & tests[b])
            print(f'    {exp(a)} / {exp(b)}: {len(ov)} seeds ({ov[0]}..{ov[-1]})')
if not found:
    print('    none')
raise SystemExit(1 if bad else 0)
