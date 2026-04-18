"""Setup configuration for the Shawwaf package."""

from setuptools import setup, find_packages

setup(
    name="shawwaf",
    version="0.1.0",
    description="A from-scratch Python image-processing library emulating a subset of OpenCV.",
    author="CSE480 Student",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    ],
)
