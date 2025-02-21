import os
from setuptools import setup, find_packages

setup(
    name='power_law_rf',  # **IMPORTANT:**  Use the same name as your package directory!
    version='0.1.0',          # Start with a version number like 0.1.0
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[
    ],
    author='Elliot Paquette, Courtney Paquette, Katie Everett',        # Your name
    author_email='elliot.paquette@mcgill.ca', # Your email
    description='Power Law Random Features code', # Short description
    long_description=open('README.md').read() if os.path.exists('README.md') else '', # Long description from README.md
    long_description_content_type='text/markdown', # If using Markdown for README
    url='https://github.com/LosVolterrosHermanos/Experimental', # URL to your repo
    license='Apache 2.0',
)
