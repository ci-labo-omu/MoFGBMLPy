import os
import shutil


def clean():
    extensions = ['.pyd', '.c', '.cpp', 'html']

    try:
        shutil.rmtree("build")
        print(f"Deleted build folder")
    except OSError as e:
        print(f"Error deleting build folder: {e}")

    for dir_path, _, filenames in os.walk("src"):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                file_path = os.path.join(dir_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    clean()
