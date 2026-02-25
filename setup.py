#!/usr/bin/env python
import os
from distutils.core import setup
setup_dir = os.path.dirname(__file__)

setup(name='pydx',
      version='0.1',
      description='Database structures and utilities for maintenance of ContaminantsDB',
      author='Matthew Matlock',
      author_email='mmatlock@wustl.edu',
      url='https://www.github.com/mkmatlock/ContaminantsDB/',
      install_requires=[pkg.strip() for pkg in open(os.path.join(setup_dir, 'requirements.txt')).read().splitlines()],
      packages=['pydx'])