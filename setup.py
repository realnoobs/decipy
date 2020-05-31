#!/usr/bin/env python

import os
from setuptools import setup
from decipy import __version__


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='decipy',
    version=__version__,
    description='Multi criteria decision making with python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    maintainer='Rizki Sasri Dwitama',
    maintainer_email='sasri.project@gmail.com',
    license="MIT",
    url='https://github.com/sasriawesome/decipy',
    packages=[
        'decipy',
    ],
    include_package_data=True,
    install_requires=[
        'pandas>=1.0',
        'matplotlib>=3.2',
        'scipy>=1.4',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    tes_suite="tests.run_tests.run_tests"
)
