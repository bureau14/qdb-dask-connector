## Project Structure for Code Navigation

- `qdb/`: QuasarDB dependencies: header files, libraries and utilities. Not part of this git repo, never change.
  - `include/qdb/`: C API header files that define types and functions that CGO integrates directly with.
- `qdb_dask_connector/`: Python source code files
- `/scripts`: Utility scripts: never change.
  - `/tests/setup/`: Git submodule that defines scripts for starting and stopping the QuasarDB daemon in the background. Do **not** modify these scripts; syncing the submodule to a newer revision is allowed.
  - `/teamcity/`: Scripts invoked by TeamCity, our CI/CD tool
    - `10.test.sh`: Runs the tests. Expects QuasarDB daemon to run in the background.
 - `/codex/`: Scripts used by OpenAI Codex to download dependencies and prepare the environment.
