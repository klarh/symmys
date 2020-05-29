#!/usr/bin/env python

import os
from setuptools import setup

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

version_fname = os.path.join(THIS_DIR, 'symmys', 'version.py')
with open(version_fname) as version_file:
    exec(version_file.read())

readme_fname = os.path.join(THIS_DIR, 'README.md')
with open(readme_fname) as readme_file:
    long_description = readme_file.read()

setup(name='symmys',
      author='Matthew Spellings',
      author_email='matthew.p.spellings@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      description='Library for identifying symmetries in point clouds',
      entry_points={},
      extras_require={},
      install_requires=[
          'numpy',
          'tensorflow',
      ],
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=[
          'symmys',
          'symmys.layers',
      ],
      python_requires='>=3',
      version=__version__
      )
