from setuptools import setup, find_packages

setup(
    name='text-clustering',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='Apache 2.0',
    description='Text clustering',
    long_description=open('README.md').read(),
    install_requires=[],
    url='https://github.com/psteitz/text-clustering',
    author='Phil Steitz',
    author_email='phil@steitz.com'
)
