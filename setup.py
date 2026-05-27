from __future__ import print_function
from setuptools import setup
from setuptools.command.test import test as TestCommand
import codecs
import os
import sys
import re

here = os.path.abspath(os.path.dirname(__file__))


# Get the version number from _version.py, and exe_path (learn from tombo)
verstrline = open(os.path.join(here, 'pyrameth', '_version.py'), 'r').readlines()[-1]
vsre = r"^PYRAMETH_VERSION = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "pyrameth/_version.py".')


# def find_version(*file_paths):
#     version_file = read(*file_paths)
#     version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
#                               version_file, re.M)
#     if version_match:
#         return version_match.group(1)
#     raise RuntimeError("Unable to find version string.")


# class PyTest(TestCommand):
#     def finalize_options(self):
#         TestCommand.finalize_options(self)
#         self.test_args = ['--strict', '--verbose', '--tb=long', 'tests']
#         self.test_suite = True
#
#     def run_tests(self):
#         import pytest
#         errno = pytest.main(self.test_args)
#         sys.exit(errno)


def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()


long_description = read('README.md')

with open('requirements.txt', 'r') as rf:
    required = rf.read().splitlines()

setup(
    name='pyrameth',
    packages=['pyrameth', 'pyrameth.utils'],
    keywords=['methylation', 'nanopore', 'neural network', 'deep learning', '5mC'],
    version=__version__,
    url='https://github.com/PengNi/pyrameth',
    download_url='https://github.com/PengNi/pyrameth/archive/{}.tar.gz'.format(__version__),
    license='BSD 3-Clause Clear License',
    author='Peng Ni',
    install_requires=required,
    python_requires='>=3.12',
    author_email='543943952@qq.com',
    description='A deep-learning method for detecting DNA methylation state '
                'from Oxford Nanopore sequencing reads (modelMTM / ModelBiLSTM)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'pyrameth=pyrameth.pyrameth:main',
            ],
        },
    platforms='any',
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
)
