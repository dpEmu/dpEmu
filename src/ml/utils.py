import json
import subprocess
from os.path import join

from src.utils import generate_unique_path, get_project_root


def run_clustering_model_script(script_name, params):
    _run_ml_script(script_name, "model", params)


def run_clustering_analyzer_script(params):
    _run_ml_script("clustering", "analyzer", params)


def _run_ml_script(script_name, script_type, params):
    path_to_params = generate_unique_path("tmp", "json")
    with open(path_to_params, "w") as file:
        json.dump(params, file)
    path_to_script = join(get_project_root(), "src/ml/{}_{}.py".format(script_name, script_type))
    _run_script("python {} {}".format(path_to_script, path_to_params).split())


def _run_script(args):
    prog = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = prog.communicate()
    if out:
        print(out)
    if err:
        print(err)
