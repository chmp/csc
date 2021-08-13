#%% Parse args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--value", type=int)

args = parser.parse_args()

#%% Write file
with open("foo.txt", "wt") as fobj:
    fobj.write("bar")
