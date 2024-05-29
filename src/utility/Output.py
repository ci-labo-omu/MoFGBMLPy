import os


class Output:
    @staticmethod
    def make_dir(path, dir_name):
        path = os.path.join(path, dir_name)
        if not os.path.exists(path):
            os.makedirs(path)