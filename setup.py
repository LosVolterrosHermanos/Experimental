
import os
from setuptools import setup, find_packages

setup(
    name='power_law_rf',
    version='0.1.1',  # Increment version to force update
    packages=['power_law_rf', 'power_law_rf.moe_plrf'],  # Explicitly list packages
    # Alternative: If find_packages still doesn't work, be explicit:
    # packages=find_packages() + ['power_law_rf.moe_plrf'],
    install_requires=[
    ],
    author='Elliot Paquette, Courtney Paquette, Katie Everett',
    author_email='elliot.paquette@mcgill.ca',
    description='Power Law Random Features code',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/LosVolterrosHermanos/Experimental',
    license='Apache 2.0',
)
