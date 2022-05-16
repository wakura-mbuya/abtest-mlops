#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['pytest==7.1.1']

test_requirements = ['pandas', 'matplotlib', 'sklearn',
                     'streamlit', 'sql', 'pytest>=3', ]

setup(
    author="",
    email="",
    python_requires='>=3.6',
    description="AB hypothesis testing",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='hypothesis_testing, Data_Preprocessing, unit_testing, pytest',
    name='AB_hypothesis_testing',
    packages=find_packages(include=['src', 'src.*']),
    test_suite='Tests',
    tests_require=test_requirements,
    url='https://github.com/abtesting10academy/abtest-mlops',
    version='0.1.0',
    zip_safe=False,
)
