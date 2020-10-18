import argparse
import sys
import os
import re

def simplify(url):
    url = re.split("[/\\\]", url)
    return [_ for _ in url if len(_) == 33][0]


def main():
    args = sys.argv[1:]
    args = [simplify(x) if 'http' in x else x for x in args]
    print('gdrive '+' '.join(args))
    # os.system('gdrive '+' '.join(args)) 
