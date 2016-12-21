#! /usr/bin/env python


from distutils.core import setup


setup(
    name='ulm',
    version='0.1',
    description='Unsupervised learning of morphology experiments',
    author='Judit Acs',
    author_email='judit@sch.bme.hu',
    packages=['ulm'],
    package_dir={'': '.'},
    provides=['ulm'],
)
