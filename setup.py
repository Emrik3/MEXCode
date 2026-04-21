from setuptools import find_packages, setup

setup(
    name="MEXCODE",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "matplotlib",
        "numpy",
        "pathlib",
    ],
)
