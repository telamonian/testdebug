#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name='testdebug',
    description='Set up Python unit tests that can be easily debugged.',
    long_description="""This package provides `TestBase`, a drop-in replacement
for `unittest.TestCase`. Normally, Python's `unittest` framework takes control
of test execution at a deep level. This can interfere with various debugging
tools, and so prevent breakpoints from taking effect. `TestBase` avoids these
problems.
""",
    version='1.0.0',

    author='Max Klein',
    url='https://github.com/telamonian/testdebug',
    license='BSD',
    platforms="Linux, Mac OS X, Windows",
    keywords=['Testing', 'Debugging', 'Profiling', 'Numpy'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],

    install_requires=[
        'numpy',
        'six',
    ],
    extras_require={
        'optional': ['argcomplete', 'pytest'],
    },

    packages=find_packages(),
)
