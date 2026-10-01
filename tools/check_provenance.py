"""Provenance checks, run from the repository root: python tools/check_provenance.py

1. Code hashes: every 'file (sha256 xxxx)' quoted in preregistrations/*.md is recomputed, first in original-fr/
   (files as executed for experiments 2 to 7), then in src/ (experiments 8 and 9).
2. Seed ledger: which seeds appear in which result files, and every overlap between test seeds of different
   experiments (development and tuning files are listed separately).
"""
import glob, hashlib, json, os, re, collections

def sha16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]

print('1. Code hashes quoted in preregistrations')
bad = 0
for pre in sorted(glob.glob('preregistrations/*.md')):
    for name, h in re.findall(r'([\w/]+\.py) \(sha256 ([0-9a-f]{16})\)', open(pre).read()):
        base = os.path.basename(name)
        found = [p for p in ('original-fr/' + base, 'src/' + base) if os.path.exists(p) and sha16(p) == h]
        bad += not found
        print(f'  {os.path.basename(pre):28s} {base:24s} {h}  ' + (f'OK ({found[0]})' if found else 'MISMATCH'))
print(f'  -> {"all hashes match" if bad == 0 else str(bad) + " mismatch(es)"}\n')

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
