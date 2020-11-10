#!/usr/bin/env python3
import sys
import os

def rep_to_float(inputfile, k, outputfile):
    with open(outputfile, 'w') as out:
        for line in open(inputfile):
            stripped_line = line.strip()
            if len(stripped_line) != 0:
                split_data = stripped_line.split(' ')
                out.write(' '.join(str(int(val) / 2**k) for val in split_data))
            out.write('\n')
if __name__ == "__main__":
    try:
        inputfile = str(sys.argv[1])
    except:
        print('Usage: python3 {} [INPUT FILE NAME] [K EXPONENT] [OUTPUT FILE NAME]'.format(sys.argv[0]))
        sys.exit(1)

    try:
        k = int(sys.argv[2])
    except:
        k = 12

    try:
        outputfile = str(sys.argv[3])
    except:
        outputfile = 'Player-Data/Input-P0-0'
        os.makedirs(os.path.dirname(outputfile), exist_ok = True)

    rep_to_float(inputfile, k, outputfile)
