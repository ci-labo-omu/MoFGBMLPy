import csv
import math
import os


def get_datasets(datasets_dir="../dataset"):
    datasets = {}
    for folder in os.listdir(datasets_dir):
        datasets[folder] = []
        for items in os.walk(os.path.join(datasets_dir, folder)):
            if "subdata" in items[0].split(os.sep):
                continue
            files = []
            for file in items[2]:
                path = os.path.join(items[0], file)
                try:
                    with open(path, newline='') as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        if len(header) < 3:
                            raise Exception("Invalid header")
                        for i in range(3):
                            int(header[i])
                    files.append(path)
                except: pass  # Invalid format (not csv or invalid header)

            datasets[folder] += files
    return datasets
