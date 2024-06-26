"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="change_detection",
    version="1.0.0",  # required
    author="Jacob Buffa", # optional
    author_email="jacobbuffa10@gmail.com",
    maintainer = "Jacob Buffa", #optional
    maintainer_email="jacobbuffa10@gmail.com",
    description="Package used to identify changes in time series data",
    packages=['change_detection'],
    long_description=long_description,  # optional
    long_description_content_type="text/markdown",  # 0ptional (see note above)
    install_requires = ["numpy", "jax", "numpyro"]
)
