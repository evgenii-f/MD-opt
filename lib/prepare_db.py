import os, sys
from pathlib import Path
SUBDIRS = ['pic', 'grid']

def prepare_dmdir(dirpath=None):
    if dirpath is None:
        dirpath = os.getcwd()
    if not os.path.exists(dirpath):
        print(f"Database dir {dirpath} was not found and is created")
    for sd in SUBDIRS:
        sd_path = os.path.join(dirpath, sd)
        if not os.path.exists(sd_path):
            os.makedirs(sd_path)
        else:
            print(f"Warning! Directory {sd_path} already exists. The original directory is not modified.")

if __name__ == "__main__":
    dirpath = (sys.argv[1])
    prepare_dmdir(dirpath)