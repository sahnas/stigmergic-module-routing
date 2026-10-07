import subprocess, time, os, glob, sys, json
t0 = time.time()
def run(cmd, timeout=20000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr; print(out[:1500] + ('\n...\n' + out[-5000:] if len(out) > 6500 else out[1500:]), flush=True)
    if r.returncode != 0: raise SystemExit(f"STEP FAILED (exit {r.returncode}): {cmd}")
h5 = glob.glob("/kaggle/input/**/tworoom.h5", recursive=True); rt = glob.glob("/kaggle/input/**/tworoom_retrained/weights.pt", recursive=True)
if not h5 or not rt: raise SystemExit(f"INPUTS MISSING h5={h5} retrained={rt}")
H = "/kaggle/working/stablewm"; os.environ.update(STABLEWM_HOME=H, WANDB_MODE="disabled", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
os.makedirs(f"{H}/datasets", exist_ok=True); os.makedirs(f"{H}/checkpoints", exist_ok=True)
os.symlink(h5[0], f"{H}/datasets/tworoom.h5"); os.symlink(os.path.dirname(rt[0]), f"{H}/checkpoints/tworoom_retrained")
run("nvidia-smi --query-gpu=name --format=csv,noheader; apt-get install -y -qq swig libegl1 libgl1 > /dev/null 2>&1; echo apt ok")
run("pip install 'stable-worldmodel[train,format]==0.1.1' 'transformers<5' hydra-core pygame pymunk shapely opencv-python-headless 2>&1 | tail -1")
run("python -c 'import hydra, stable_worldmodel, pygame, pymunk, shapely, cv2; print(\"imports ok\")'")
run("git clone https://github.com/lucas-maes/le-wm.git /kaggle/tmp/le-wm > /dev/null 2>&1; cd /kaggle/tmp/le-wm && git checkout -q 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac && git log -1 --format='le-wm %h %cd' --date=short")
# instrument eval.py: log, at every policy call, the agent and target positions, the distance to target, the terminated flags,
# the action returned, and at every replanning the predicted cost of the chosen plan
patch = r'''
import json as _json, numpy as _np, torch as _torch
_LOG = {"steps": [], "plans": []}
def _wrap(policy, tag):
    solver = policy.solver; orig_call = solver.__call__
    def logged_call(info_dict, **kw):
        out = orig_call(info_dict, **kw)
        c = out.get("costs"); c = [float(x) for x in (c if isinstance(c, list) else [])]
        _LOG["plans"].append({"n_envs": int(out["actions"].shape[0]), "predicted_costs": c[-int(out["actions"].shape[0]):] if c else []})
        return out
    solver.__call__ = logged_call
    orig_get = policy.get_action
    def logged_get(info_dict, **kw):
        rec = {}
        for k in ("pos_agent", "pos_target", "distance_to_target", "terminated", "goal_pos_agent"):
            if k in info_dict:
                v = info_dict[k]; rec[k] = _np.asarray(v).tolist()
        a = orig_get(info_dict, **kw); rec["action"] = _np.asarray(a).tolist(); _LOG["steps"].append(rec); return a
    policy.get_action = logged_get
    return policy
'''
code = open("/kaggle/tmp/le-wm/eval.py").read()
code = code.replace("import stable_worldmodel as swm", "import stable_worldmodel as swm\n" + patch, 1)
code = code.replace("    world.set_policy(policy)", "    if cfg.get('policy', 'random') != 'random':\n        policy = _wrap(policy, cfg.policy)\n    world.set_policy(policy)", 1)
code = code.replace('    print(metrics)', '    print(metrics)\n    _json.dump(_LOG, open(f"/kaggle/working/log_{str(cfg.policy).replace(\'/\', \'_\')}.json", "w"))', 1) if '    print(metrics)' in code else code + '\n'
open("/kaggle/tmp/le-wm/eval_logged.py", "w").write(code)
assert "_wrap(policy" in code and "_json.dump" in code, "patch failed"
for pol, tag in [("tworoom_retrained", "retrained"), ("quentinll/lewm-tworooms", "released")]:
    run(f"cd /kaggle/tmp/le-wm && python eval_logged.py --config-name=tworoom.yaml policy={pol} eval.num_eval=100 output.filename=/kaggle/working/{tag}_results.txt 2>&1 | grep -vE 'arn' | tail -6", timeout=12000)
run("ls -la /kaggle/working/*.json; python -c \"import json,glob; [print(f, len(json.load(open(f))['steps']), 'steps', len(json.load(open(f))['plans']), 'plans') for f in glob.glob('/kaggle/working/log_*.json')]\"")
print(f"\nTOTAL {time.time()-t0:.0f}s")
