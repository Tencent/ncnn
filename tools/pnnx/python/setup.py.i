import sys
from setuptools import setup, find_packages

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

except ImportError:
    bdist_wheel = None

if sys.version_info < (3, 0):
    sys.exit("Sorry, Python < 3.0 is not supported")

requirements = ["torch"]

setup(
        name="pnnx",
        version="${PACKAGE_VERSION}",
        author="nihui",
        author_email="nihuini@tencent.com",
        description="pnnx is an open standard for PyTorch model interoperability.",
        url="https://github.com/Tencent/ncnn/tree/master/tools/pnnx",
        classifiers=[
            "Programming Language :: C++",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
        ],
        license="BSD-3",
        python_requires=">=3.6",
        packages=find_packages(),
        package_dir={"": "."},
        package_data={"pnnx": ["pnnx${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}"]},
        install_requires=requirements,
        cmdclass={"bdist_wheel": bdist_wheel},
)