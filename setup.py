# encoding: utf-8
#
# Copyright (c) 2019 Glencoe Software, Inc. All rights reserved.
#
# This software is distributed under the terms described by the LICENCE file
# you can find at the root of the distribution bundle.
# If the file is missing please request a copy by contacting
# support@glencoesoftware.com.

import os
import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

import version

# Hack to prevent stupid "TypeError: 'NoneType' object is not callable" error
# in multiprocessing/util.py _exit_function when running `python
# setup.py test` or `python setup.py flake8`.  See:
#  * http://www.eby-sarna.com/pipermail/peak/2010-May/003357.html)
#  * https://github.com/getsentry/raven-python/blob/master/setup.py
import multiprocessing
assert multiprocessing  # silence flake8


def get_requirements(suffix=''):
    with open('requirements%s.txt' % suffix) as f:
        rv = f.read().splitlines()
    return rv


class PyTest(TestCommand):

    user_options = [('pytest-args=', 'a', 'Arguments to pass to py.test')]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        if isinstance(self.pytest_args, str):
            # pytest requires arguments as a list or tuple even if singular
            self.pytest_args = [self.pytest_args]
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


def read(fname):
    """
    Utility function to read the README file.
    :rtype : String
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='isyntax2raw',
      version=version.getVersion(),
      python_requires='>=3.6',
      description='iSyntax to raw format converter',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      classifiers=[],  # Get strings from
                       # http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='',
      author='Glencoe Software, Inc.',
      author_email='info@glencoesoftware.com',
      url='https://github.com/glencoesoftware/isyntax-to-rst',
      license='License :: OSI Approved :: BSD License',
      packages=find_packages(),
      package_data={'isyntax2raw': ['resources/*.xml']},
      zip_safe=True,
      include_package_data=True,
      platforms='any',
      setup_requires=['flake8'],
      install_requires=[
          'click==7.0',
          'pillow>=7.1.0',
          'numpy==1.17.3',
          'zarr==2.8.1',
          'kajiki==0.8.2',
          'fsspec>=0.9.0',
      ],
      tests_require=[
          'flake8',
          'pytest',
      ],
      cmdclass={'test': PyTest},
      entry_points={
          'console_scripts': [
              'isyntax2raw = isyntax2raw.cli.isyntax2raw:main',
          ]
      }
      )
