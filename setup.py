"""Setup script for creating package."""
from setuptools import setup, find_packages
from dlh_utils._version import __version__

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
        
setup(
    name='dlh_utils',
    version=__version__,
    description="pyspark pipeline function package from Data Linkage Hub team",
    author="David Cobbledick, Madalina Iova, Jenna Hart & Anthony Edwards",
    packages= [
      'dlh_utils'
    ],
    install_requires=requirements
)