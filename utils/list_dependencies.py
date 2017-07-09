# -*- coding: utf-8 -*-

# This utility lists the dependencies of a package

from pip._vendor import pkg_resources
import sys

def get_deps(pn):
    _package_name = pn
    _package = pkg_resources.working_set.by_key[_package_name]
    return _package.requires()
    print([str(r) for r in _package.requires()])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit()
    
    print(sys.argv[1])
    list_deps(sys.argv[1])