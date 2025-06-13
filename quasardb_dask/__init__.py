import dask

# Avoids a lot of common pitfalls with unaligned datasets
dask.config.set({"dataframe.shuffle.method": "tasks"})

from .client_side import *
