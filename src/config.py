#!/usr/bin/python

import yaml, sys

def config(confFile):
    try:
        fiConfs = open(confFile, "r")
    except:
        print("Unable to find/open the config file in:", confFile)
        print("Make sure the correct path is passed to the inject call.")
        sys.exit()
    if(confFile.endswith(".yaml")):
        fiConf = yaml.safe_load(fiConfs)
    else:
        print("Unsupported file format:", confFile)
        sys.exit()
    return fiConf