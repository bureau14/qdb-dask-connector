#!/usr/bin/env bash
set -u -x

PYTHON="${PYTHON_CMD:-python3}"


if [[ -d "dist/" ]]
then
    echo "Removing dist/"
    rm -rf dist/
fi

if [[ -d ".env/" ]]
then
    echo "virtualenv already exists, removing it"
    rm -rf .env/
fi

echo "Creating new virtualenv"
${PYTHON} -m venv --clear ./.env

if [[ "$(uname)" == MINGW* ]]
then
    VENV_PYTHON=".env/Scripts/python.exe"
else
    VENV_PYTHON=".env/bin/python"
fi

# install requirements

${VENV_PYTHON} -m pip install --upgrade -r dev-requirements.txt
${VENV_PYTHON} -m pip install --no-deps --force-reinstall qdb/quasardb-*.whl

# build wheel

echo "Building wheel"

${VENV_PYTHON} -m build --wheel

# install wheel
echo "Installing built wheel"

${VENV_PYTHON} -m pip install --no-deps --force-reinstall dist/quasardb_dask*.whl


echo "Invoking pytest"

TEST_OPTS="$@"
if [[ ! -z ${JUNIT_XML_FILE-} ]]
then
    TEST_OPTS+=" --junitxml=${JUNIT_XML_FILE}"
fi

exec ${VENV_PYTHON} -m pytest ${TEST_OPTS}
