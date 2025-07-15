from setuptools import setup, find_packages

setup(
    name="algoview",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "seaborn>=0.11.0"
    ],
) 