# qdb-dask-connector
Integration between QuasarDB's Python API and Dask


## Build and test locally

### Prerequisites

* Python >= 3.9

The instructions below have been verified to work on:
* Linux (Ubuntu, WSL Ubuntu, Debian);
* macOS 15.4.1 (amd64 and aarch64);
* Windows;

### QuasarDB tarball extraction

All QuasarDB APIs assume QuasarDB and associated utilities are extracted into the `qdb/` subdirectory.

Extract QuasarDB C API, utilities and server into qdb/
```
mkdir qdb
cd qdb
tar xf <archives>
cd ..
```

### QuasarDB Python API extraction

You should provide QuasarDB Python API wheel compatible with your platform to qdb/ directory.

### Launch services

Use the scripts from the qdb-test-setup submodule to start and stop background services. These scripts are used across all QuasarDB API and tools projects:

```
$ scripts/tests/setup/start-services.sh

<.. snip ..>

qdbd secure and insecure were started properly.
```

## Run tests

Invoke the scripts that our continuous integration system uses directly:

```
$ bash scripts/teamcity/10.test.sh

<.. snip a lot ..>

========================================================================================= 650 passed, 0 skipped, 40 warnings in 87.43s (0:01:27) ==========================================================================================
$

```

This does the following out of the box:

* Create a virtualenv;
* Install dev requirements in virtualenv;
* Build the .whl file;
* Install the .whl file in virtualenv;
* Invoke pytest on the entire repository in the `tests/` subdirectory.

All arguments passed to this `10.test.sh` script are passed directly to pytest. For example, to enable verbose output and test a single file, you can use this:

```
$ bash scripts/teamcity/10.test.sh -s tests/test_dask_integration.py

<.. snip a lot ..>

(.env) igorn@igor-desktop:~/qdb-dask-connector$ pytest -s tests/test_dask_integration.py
============================================== test session starts ===============================================
platform linux -- Python 3.12.3, pytest-8.3.5, pluggy-1.6.0
rootdir: /home/igorn/qdb-dask-connector
configfile: pyproject.toml
collected 46 items                                                                                               

tests/test_dask_integration.py ..............................................

========================================= 46 passed, 5 warnings in 4.63s =========================================
```
