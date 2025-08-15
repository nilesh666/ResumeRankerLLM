from setuptools import setup, find_packages

with open("requirements.txt") as f:
    req = f.read().splitlines()

setup(
    name="Resume-Ranker",
    version="0.1",
    author="Nilesh",
    packages=find_packages(),
    install_requires=req
)