# -*- coding: utf-8 -*-
import inspect
import os
import parser
import ntpath
import configparser
import argparse
import subprocess
import time
from subprocess import Popen

# config
RUNTIME_ENABLE = False
RUNTIME_N_LOOP = 50

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def cprint(*args, c=bcolors.OKBLUE):
    print(c + " ".join(map(str,args)) + bcolors.ENDC)
def print_head(*args):
    cprint(*args, c=bcolors.OKBLUE + bcolors.BOLD)
def print_ok(*args):
    cprint(*args, c=bcolors.OKGREEN)
def print_err(*args):
    cprint(*args, c=bcolors.FAIL + bcolors.BOLD)
def print_imp(*args):
    cprint(*args, c=bcolors.OKCYAN + bcolors.BOLD)

DATA_PATH = "./test_data/"
OPTIONS_FILE = "options.ini"

if __name__ == "__main__":
    # load everything necessary
    argparser = argparse.ArgumentParser(description='Test or simply run the simulators on data.')
    argparser.add_argument('of', metavar='options_file', type=str,
                            help='path relative to ./test_data/ containing the command line options for this run')
    argparser.add_argument('-m', action='store_true',
                            help='obtain manual i/o control of the simulator after start')
    args = argparser.parse_args()
    print_head("Setup:")

    here = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    print("Changing path to '" + here + "'.")
    os.chdir(here)

    if (not os.path.isfile(DATA_PATH + args.of)):
        print_err("Options not found at " + DATA_PATH + args.of + ". Aborting.")
        quit()
    config = configparser.ConfigParser()
    config.optionxform=str
    config.read(DATA_PATH + args.of)
    print("Found options file: '" + args.of + "'.")

    # parse the options and create cmdline
    if (not os.path.isfile(config["SIMULATOR"]["path"])):
        print_err("- Simulator not found. Aborting.")
        quit()
    print("- Found simulator")
    mandatory = 0
    optional = 0
    opts = []
    for k, v in config.items("OPTIONS"):
        opts += [k + "=" + v.replace('""', '"') + ""]
        if (k == "a_path" or k == "assignment" or k == "s_path" or k == "structure" or k == "lib_path"):
            mandatory += 1
        else:
            optional += 1
    if (mandatory != 3):
        print_err("- Structure or assignment missing. Aborting.")
        quit()
    print("- Found " + str(mandatory) + " + " + str(optional) + " cmdline options")
    print(opts)

    # run the simulation
    print_head("\nSummary:")
    print("- Simulator: ")
    print_ok("   " + ntpath.basename(config["SIMULATOR"]["path"]))
    print("- Options:")
    for opt in opts:
        print_ok("   " + opt)

    print_head("\nRun:")
    if (args.m == True):
        with Popen(["python"] + [config["SIMULATOR"]["path"]] + opts) as sim:
            pass
    else:
        with Popen(["python"] + [config["SIMULATOR"]["path"]] + opts, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as sim:
            commands = ''
            n_loop = 1
            if (RUNTIME_ENABLE == True):
                n_loop = RUNTIME_N_LOOP
            for n in range(n_loop):
                commands += 'start\n'
            start = time.time()
            strs = str(sim.communicate(input=(commands + 'exit').encode('utf-8'))[0].decode("utf-8"))
            end = time.time() - start
            for line in strs.splitlines():
                if line[0] == 'O':
                    print_imp(line)
                else:
                    print(line)
            if (RUNTIME_ENABLE == True):
                print("Runtime (" + str(RUNTIME_N_LOOP) + "): " + str(end))
            print("")
