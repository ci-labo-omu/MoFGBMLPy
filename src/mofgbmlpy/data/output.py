import os
import csv


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
    def save_results(results_data, path):
        fields = list(results_data[0].keys())

        with open(path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results_data)
