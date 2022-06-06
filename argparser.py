#!/usr/bin/python3
import os
import subprocess

import argparse as argp

def create_parser():
    parser = argp.ArgumentParser(description="")

    parser.add_argument('-d', '--data-dir', default='.', help='directory where images contains')
    parser.add_argument('-f', '--filename', default=None, type=str, required=True, dest='filename', help='image filename. leave None if you want to apply for all files in directory')
    parser.add_argument('-c', '--clahe', default=True, type=bool, dest='clahe', help='Use local histogram equalization if true')
    parser.add_argument('-v', '--verbose', default=False, type=bool, dest='verbose', help='Show interim result in napari')
    parser.add_argument('--sigma-start', default=0.5, type=float, dest='start', help='')
    parser.add_argument('--sigma-end', default=4, type=float, dest='end', help='')
    parser.add_argument('--sigma-step', default=1, type=float, dest='step', help='')
    parser.add_argument('-o', '--output', default='./output/', type=str, dest='output_dir', help='')
    parser.add_argument('-p', '--prefix', default='', type=str, dest='prefix', help='Prefix for output files')
    parser.add_argument('-s', '--suffix', default='', type=str, dest='suffix', help='Suffix for output files')
    return parser

# def check_args(ns):
#     for name, files in vars(ns).items():
#         if type(files) is list:
#             for file in files:
#                 if not os.path.exists(file):
#                     print("ERROR! File {} not found!".format(file))
#                     return False
#         elif files != None and name != 'save_file':
#             if not os.path.exists(files):
#                 print("ERROR! File {} not found!".format(files))
#                 return False
#     return True