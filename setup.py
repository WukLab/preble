from setuptools import setup, find_packages
import os

def read_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name="preble",
    version="0.1.01",
    author="Vikranth Srivatsa",
    author_email="vsrivatsa@ucsd.edu",
    description="load balancer",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wuklab/preble",
    packages=find_packages(include=["preble", "preble.*"]),
    include_package_data=True,
    package_data={
        '': ['assets/*.jpg', 'docs/*.md'],
    },
    install_requires=[
        'scipy',
        'matplotlib',
        'datasets',
        'paramiko',
        'fire',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'preble=preble.server.server:main',
        ],
    },
)
