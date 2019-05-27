import subprocess


def run_ml_script(cline):
    _run_script(cline.split())


def _run_script(args):
    prog = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = prog.communicate()
    if out:
        print(out)
    if err:
        print(err)
