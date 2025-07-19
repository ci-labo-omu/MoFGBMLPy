from setuptools import setup, find_packages
import pathlib
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy
import os
import sys
import glob

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

# Function to get all C++ source files from core directory
def get_cpp_sources():
    cpp_files = []
    for pattern in ['src/mofgbmlpy/core/**/*.cpp']:
        cpp_files.extend(glob.glob(pattern, recursive=True))
    return cpp_files

# Get all C++ source files
cpp_sources = get_cpp_sources()

cython_files = []
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.pyx'):
            path = os.path.join(root, file)
            path_without_extension = ".".join(path.split(".")[:-1])
            name = ".".join(path_without_extension.split(os.sep)[1:])

            # Add core C++ sources to each extension
            sources = [path] + cpp_sources

            cython_files.append(Extension(name,
                                          sources,
                                          extra_compile_args=[openmp_arg, optimization_arg, cpp_std_arg],
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
