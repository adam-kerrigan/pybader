import os
from pkg_resources import iter_entry_points, resource_listdir
from pybader.dunders import (
        __name__,
        __version__,
        __author__,
        __desc__,
        __long_desc__,
        __config__,
)

__doc__ = __desc__ + '\n\n' + __long_desc__
del(__desc__, __long_desc__)

# check for the config file and write if not found
if not os.path.isfile(__config__):
    print("  First run initialisation:")
    func_map = {}
    print("  Grouping entry points: ", end='')
    for ep in iter_entry_points(group='pybader_ep'):
        func_map.update({ep.name: ep.load()})
    print("Done.")
    func_map['create_config']()
    # If no cached numba functions exits cache them
    recache = True
    for cache in resource_listdir('pybader', '__pycache__'):
        if any(cache[-3:] == ext for ext in ['nbc', 'nbi']):
            recache = False
    if recache:
        func_map['cache_JIT']()
