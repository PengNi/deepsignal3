from __future__ import print_function
from setuptools import setup
from setuptools.command.test import test as TestCommand
import codecs
import os
import sys
import re

here = os.path.abspath(os.path.dirname(__file__))


# Get the version number from _version.py, and exe_path (learn from tombo)
verstrline = open(os.path.join(here, 'deepsignal3', '_version.py'), 'r').readlines()[-1]
vsre = r"^DEEPSIGNAL3_VERSION = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "deepsignal3/_version.py".')


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


long_description = read('README.rst')

with open('requirements.txt', 'r') as rf:
    required = rf.read().splitlines()

setup(
    name='deepsignal3',
    packages=['deepsignal3', 'deepsignal3.utils'],
    keywords=['methylation', 'nanopore', 'neural network'],
    version=__version__,
    url='https://github.com/PengNi/deepsignal3',
    download_url='https://github.com/PengNi/deepsignal3/archive/{}.tar.gz'.format(__version__),
    license='BSD 3-Clause Clear License',
    author='Peng Ni',
    # tests_require=['pytest'],
    install_requires=required,
    # cmdclass={'test': PyTest},
    author_email='543943952@qq.com',
    description='A deep-learning method for detecting DNA methylation state '
                'from Oxford Nanopore sequencing pore-c reads',
    long_description=long_description,
    entry_points={
        'console_scripts': [
            'deepsignal3=deepsignal3.deepsignal3:main',
            ],
        },
    platforms='any',
    # test_suite='test',
    zip_safe=False,
    include_package_data=True,
    # package_data={'deepsignal3': ['utils/*']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        ],
    # extras_require={
    #     'testing': ['pytest'],
    #   },
)
