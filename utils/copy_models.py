import argparse
import sys, os

sys.path = [os.getcwd()] + sys.path

from utils.tools import dict_filter, get_targets, get_hparams, get_artifacts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name')
    parser.add_argument('folder_name')
    parser.add_argument('--path', default='/data/grid_nerf/')
    parser.add_argument('--recent', default=0, type=int)
    opt = parser.parse_args()
    path = os.path.join(opt.path, opt.folder_name)
    os.makedirs(path, exist_ok=True)
    targets = get_targets(dict_filter({'project_name': opt.project_name}))
    targets = sorted(targets, key=lambda x: os.path.getctime(x))
    if opt.recent != 0:
        targets = targets[-opt.recent:]
    for folder in targets:
        try:
            hparams = get_hparams(folder)
            artifact = get_artifacts(folder, "*.gnrf")
            subdir = os.path.join(path, "_".join(artifact.split('/')[-1].split("_")[:2]))
            os.makedirs(subdir, exist_ok=True)
            os.system(f"cp {artifact} {subdir}")
            print(f"copy from {artifact} to {subdir}")
        except:
            print(f"copying {folder} failed!")
