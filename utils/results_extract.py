import argparse
import sys, os

sys.path = [os.getcwd()] + sys.path

from utils.tools import batch_result_extract

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name')
    parser.add_argument('--name_by_id', default=False, action='store_true')
    parser.add_argument('--path', default='../artifacts')
    parser.add_argument('--artifact', default='metric.pkl')
    opt = parser.parse_args()
    batch_result_extract(path=opt.path, project_name=opt.project_name, artifact_name=opt.artifact, name_by_id=opt.name_by_id)
