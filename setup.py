from setuptools import setup, find_packages

setup(
    name='mygrad',
    version='0.0.1',
    packages=find_packages(include=['original']),
    install_requires=['numpy', 'jax', 'pytest'],
)