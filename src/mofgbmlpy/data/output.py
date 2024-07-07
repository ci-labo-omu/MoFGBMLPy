import xml.etree.cElementTree as xml_tree
import os
import csv

import numpy
import numpy as np


class Output:
    @staticmethod
    def mkdirs(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    @staticmethod
    def writeln(file_name, txt, append=False):
        with open(file_name, 'a' if append else 'w') as f:
            f.write(txt)

    @staticmethod
    def writelns(file_name, lns, append=False):
        with open(file_name, 'a' if append else 'w') as f:
            for ln in lns:
                f.write(ln)
                f.write('\n')

    @staticmethod
    def save_results(results_data, path, args=None):
        if isinstance(results_data, xml_tree.ElementTree):
            if args is not None and args.has_key("PRETTY_XML") and args.get("PRETTY_XML"):
                xml_tree.indent(results_data, space="\t", level=0)
            results_data.write(path, encoding="utf-8", xml_declaration=True)
        elif isinstance(results_data, np.ndarray):
            fields = list(results_data[0].keys())

            with open(path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results_data)
        else:
            raise TypeError("results_data must be an instance of xml.etree.ElementTree or dict")
