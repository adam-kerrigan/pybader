import os

from pkg_resources import iter_entry_points, resource_listdir

from pybader.dunders import (__author__, __config__, __desc__, __long_desc__,
                             __name__, __version__)

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
    # cache the jitable functions
    func_map['cache_JIT']()
