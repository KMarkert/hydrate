
import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='archhydro',
      version='0.0.1',
      description='ArchHydro is a Python package to setup and run hydrologic models using data from Earth Engine',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/kmarkert/archhydro',
      packages=setuptools.find_packages(),
      author='Kel Markert',
      author_email='kel.markert@gmail.com',
      license='MIT',
      zip_safe=False,
      include_package_data=True,
)
