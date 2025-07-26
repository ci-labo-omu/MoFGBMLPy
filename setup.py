from setuptools import setup, find_packages
import pathlib
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy
import os
import sys
import glob
import re

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
    optimization_arg = "/Ox"
    cpp_std_arg = "/std:c++17"
else:
    openmp_arg = '-fopenmp'
    optimization_arg = "-Ox"
    cpp_std_arg = "-std=c++17"

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")


def parse_includes_from_hpp(hpp_path):
    includes = []
    if not os.path.exists(hpp_path):
        return includes
    
    try:
        with open(hpp_path, 'r', encoding='utf-8') as f:
            content = f.read()
            local_includes = re.findall(r'#include\s*"([^"]+)"', content)
            includes.extend(local_includes)
    except Exception as e:
        print(f"Warning: Could not read {hpp_path}: {e}")
    
    return includes

def find_hpp_iterative(start_hpp_path, base_dir):
    visited = set()
    stack = [start_hpp_path]
    all_hpp_files = []

    while stack:
        hpp_path = stack.pop()

        if hpp_path in visited:
            continue
        visited.add(hpp_path)
        all_hpp_files.append(hpp_path)

        includes = parse_includes_from_hpp(hpp_path)
        current_dir = os.path.dirname(hpp_path)

        for include in includes:
            if not include.endswith('.hpp'):
                continue

            include_path = os.path.join(current_dir, include)
            if not os.path.exists(include_path):
                include_path = os.path.join(base_dir, include)
            if os.path.exists(include_path) and include_path not in visited:
                stack.append(include_path)

    return all_hpp_files


def get_cpp_sources(py_path):
    base_dir = py_path.replace("mofgbmlpy", f"mofgbmlpy{os.sep}core")
    cpp_path = base_dir + ".cpp"
    hpp_path = base_dir + ".hpp"

    sources = []

    if not os.path.exists(cpp_path) or not os.path.exists(hpp_path):
        return []

    all_hpp_files = find_hpp_iterative(hpp_path, base_dir)

    for hpp_file in all_hpp_files:
        corresponding_cpp = hpp_file.replace('.hpp', '.cpp')
        if os.path.exists(corresponding_cpp) and corresponding_cpp not in sources:
            sources.append(corresponding_cpp)

    print(f"Adding C++ sources for {py_path}: {sources}")

    return sources


cython_files = []
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.pyx'):
            path = os.path.join(root, file)
            path_without_extension = ".".join(path.split(".")[:-1])
            name = ".".join(path_without_extension.split(os.sep)[1:])

            # Add core C++ sources to each extension
            sources = [path] + get_cpp_sources(path_without_extension)

            cython_files.append(Extension(name,
                                          sources,
                                          # extra_compile_args=[openmp_arg, optimization_arg, cpp_std_arg],
                                          extra_compile_args=[cpp_std_arg],
                                          language='c++',
                                          include_dirs=[numpy.get_include(), "src/mofgbmlpy"]))


setup(
    ext_modules=cythonize(
        cython_files,
        compiler_directives={"language_level": "3", "profile": True},
        language="c++",
    ),
    name="mofgbmlpy",
    version="1.0.2",
    description="MoFGBML in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RobinMeneustOMU/MoFGBMLPy",  # Optional
    author="Robin Meneust",
    author_email="sc24209q@st.omu.ac.jp",
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src"),  # Required
    install_requires=[
        'numpy<2.0.0',
        'matplotlib<3.9',
        'scikit-learn',
        'gbml',
        'pytest',
        'Cython',
        'pyrecorder',
        'gprof2dot',
        'jproperties',
    ],  # Optional
    package_data={"mofgbmlpy.main.arguments": ["*.json"]}
)
