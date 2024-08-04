import xml.etree.cElementTree as xml_tree
import os
import csv

import numpy
import numpy as np


class Output:
    @staticmethod
    def mkdirs(dir_name):
        """Make a new directory if it doesn't exist yet

        Args:
            dir_name (string): Directory name
        """
        os.makedirs(dir_name, exist_ok=True)

    @staticmethod
    def writeln(file_name, txt, append=False):
        """Write a text in a file

        Args:
            file_name (str): Name of the file where the text will be written
            txt (str): Text to write
            append (bool): If True then append text to the end of the file otherwise overwrite existing file
        """
        with open(file_name, 'a' if append else 'w') as f:
            f.write(txt)

    @staticmethod
    def save_data(data, path, pretty_xml=None):
        """Save the given data to a file

        Args:
            data (dict | xml.etree.ElementTree): data to be saved
            path (str): Path of the file where the data will be saved
            pretty_xml (bool): If True then add indent and line breaks to the XML file content (if the data is not an XML element it's ignored)
        """
        if isinstance(data, xml_tree.ElementTree):
            if pretty_xml is not None and pretty_xml:
                xml_tree.indent(data, space="\t", level=0)
            data.write(path, encoding="utf-8", xml_declaration=True)
        elif isinstance(data, np.ndarray):
            fields = list(data[0].keys())

            with open(path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
        else:
            raise TypeError("data must be an instance of xml.etree.ElementTree or dict")
