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
    name="change_detection",  # required
    version="1.0.0",  # required
    author="Jacob Buffa", # optional
    author_email="jbuffa@astros.com", #optional
    maintainer = "Jacob Buffa", #optional
    maintainer_email="jbuffa@astros.com", #optional
    description="Package used to identify skill changes",  # optional
    packages=['change_detection'],
    long_description=long_description,  # optional
    long_description_content_type="text/markdown",  # 0ptional (see note above)
    #url="https://bitbucket.org/Jacob-Buffa/astro_kinematics",  # Optional
    install_requires = ["numpy", "jax", "numpyro"]
    # project_urls={  # Optional
    #     "Bug Reports": "https://bitbucket.org/houstonastros/astro_kinematics/issues",
    #     "Source": "https://bitbucket.org/houstonastros/astro_kinematics/",
    # },
)
