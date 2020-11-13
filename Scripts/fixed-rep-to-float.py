#!/usr/bin/env python3
import sys
import os

def rep_to_float(input_file, k, output_file):
    ifile = open(input_file, 'r')
    with open(output_file, 'w') as out:
        for line in ifile:
            stripped_line = line.strip()
            if len(stripped_line):
                split_data = stripped_line.split(' ')
                out.write(' '.join(str(int(val) / 2**k) for val in split_data))
            out.write('\n')
    ifile.close()

def main():
    try:
        input_file = str(sys.argv[1])
    except:
        print('Usage: python3 {} [INPUT FILE NAME] [K EXPONENT] [OUTPUT FILE NAME]'.format(sys.argv[0]))
        sys.exit(1)

    try:
        k = int(sys.argv[2])
    except:
        k = 12

    try:
        output_file = str(sys.argv[3])
    except:
        output_file = 'Player-Data/Input-P0-0'
        os.makedirs(os.path.dirname(output_file), exist_ok = True)

    rep_to_float(input_file, k, output_file)


if __name__ == "__main__":
    main()
