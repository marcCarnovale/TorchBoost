from pathlib import Path
import os,subprocess,time,json,signal
ROOT=Path(__file__).resolve().parents[1]
def active(script):
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=(p/'cmdline').read_bytes().split(b'\0')
            if len(a)>1 and b'python' in a[0] and a[1].decode().endswith(script):found.append(int(p.name))
        except (OSError,UnicodeError):pass
    return found
end=time.monotonic()+900
while active('unified_study.py') or active('unified_baselines.py'):
    if time.monotonic()>end:raise RuntimeError('prior block exceeded supervisor budget')
    time.sleep(2)
if not active('finish_unified.py'):
    result=subprocess.run(['python','experiments/finish_unified.py'],cwd=ROOT)
    (ROOT/'results/current/finish_exit.json').write_text(json.dumps({'returncode':result.returncode}))
