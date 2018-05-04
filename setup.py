#!/usr/bin/env python

from setuptools import setup

setup(name='caffe-generator',
      version='1.0',
      description='Generator for Caffe Models from yaml',
      author='Felix Geilert',
      author_email='f.geilert@brainplug.de',
      install_requires=[ 'fire', 'pyyaml' ],
      py_modules=['generator', ],
      entry_points={'console_scripts': [
          'caffe-generator = generator:main'
      ]})
