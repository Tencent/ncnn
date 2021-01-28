from setuptools import setup, find_packages

import sys
if sys.version_info < (3,0):
  sys.exit("Sorry, Python < 3.0 is not supported")

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

requirements = [
    "numpy",
    #"tqdm",
    #"requests",
    #"portalocker",
    #"opencv-python"
]

setup(
  name              = "ncnn",
  version           = "${PACKAGE_VERSION}",
  author            = "nihui",
  author_email      = "nihuini@tencent.com",
  maintainer        = "caishanli",
  maintainer_email  = "caishanli25@gmail.com",
  description       = "ncnn is a high-performance neural network inference framework optimized for the mobile platform",
  url               = "https://github.com/Tencent/ncnn",
  classifiers       = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
  ],
  python_requires   = ">=3.5",
  packages          = find_packages(),
  package_dir       = {"": "."},
  package_data      = {"ncnn": ["ncnn${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}"]},
  install_requires  = requirements,
  cmdclass          = {"bdist_wheel": bdist_wheel},
)
