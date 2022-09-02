

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--number")
parser.add_argument("--string")
args = parser.parse_args()
print(args.number)
print(args.string)


