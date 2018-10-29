#!/usr/bin/env python
from inspect import signature
from string import Template
from textwrap import dedent, indent
import unittest

header = dedent('''\
    class AsserterSkeleton(object):
        """Class that exposes skeletons of the 
        assert methods from unittest.TestCase
        """
''')

assertDefTemplate = Template(dedent("""
    def ${name}${sig}:
        pass
"""))

def genDefText():
    return header + ''.join([
        indent(assertDefTemplate.substitute(name=name, sig=signature(val)), prefix=' '*4)
        for name,val in vars(unittest.TestCase).items()
        if name.startswith('assert') and callable(val) and 'deprecated' not in val.__name__])

def writeSkeleton(pth):
    with open(pth, 'w') as f:
        f.write(genDefText())

def main():
    pth = '../testBase/asserterSkeleton.py'
    writeSkeleton(pth)

if __name__=='__main__':
    main()
