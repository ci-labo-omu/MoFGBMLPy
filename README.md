# MoFGBMLPy

## Install

1. Install virtualenv if you don't already have it: `python -m pip install virtualenv`
2. Create a virtual environment: `python -m venv .venv`
3. Activate this environment: `source .venv/bin/activate` (or `".venv/Scripts/activate.bat"` on Windows)
4. Install the dependencies: `python -m pip install -r requirements.txt`

`python setup.py build_ext --inplace`

### Speedup pymoo
Use the compiled version of pymoo to reduce computation time

1. `pip uninstall pymoo`
2. `mkdir pymoo`
3. `cd pymoo`
4. `git clone https://github.com/anyoptimization/pymoo`
5. `cd pymoo`
6. `make compile`
7. `pip install .`

### Note
- pymoo is not yet compatible with numpy 2.0.0, so we use the previous version instead
- Profiler for Cython only works for Python<3.12


### Profiling

#### Automatically

Run `python profiler.py MoFGBMLBasicMain` (or replace MoFGBMLBasicMain with the method you want to use)

#### Manually

1. Install `gprof2dot`
2. Generate a pstats profiler results file
3. `gprof2dot -f pstats Profile.pstats -o Profile.dot`
4. `dot Profile.dot -Tpng -o Profile.png`