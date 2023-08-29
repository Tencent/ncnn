from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.3'
DESCRIPTION = 'pnnx python wrapper'
LONG_DESCRIPTION = 'A package that allows to export simple pytorch model to pnnx/ncnn format using pnnx.'

# Setting up
setup(
    name="pnnx",
    version=VERSION,
    author="Qianshu(Ruichen Bao)",
    author_email="<ruichenbao@qq.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    package_data={
        'pnnx': ['bin/*/*']
    },
    install_requires=[],
    keywords=['python', 'ncnn', 'pnnx'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ]
)