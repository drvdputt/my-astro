#!/usr/bin/env python3
"""Open any file produced by the jwst pipeline and print out some info."""

import sys
print(sys.executable)
from jwst import datamodels
from argparse import ArgumentParser
from pprint import pprint

def main():
    ap = ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("-a", "--all", action="store_true")
    args = ap.parse_args()
    f = args.file
    with datamodels.open(f) as d:
        print(type(d))
        if args.all:
            d.info(max_rows=None)
        else:
            print("Pipeline version:")
            print(d.meta.calibration_software_version)
            print(d.meta.ref_file.crds.context_used)
            print("\nInstrument:")
            print_node_by_key(d.meta.instrument, ["detector", "channel", "band"])
            print("\nObservation:")
            print_node_by_key(d.meta.observation, ["observation_label", "bkgdtarg"])
            print("\nProgram:")
            print_node_by_key(d.meta.program, ['category', 'pi_name', 'title'])

            print(
                "\nStage 1 steps performed:",
                [key for key, value in d.meta.cal_step.items() if value == "COMPLETE"],
            )

def print_node_by_key(node, interesting_keys):
    for key, value in node.items():
        if key in interesting_keys:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
