# %%

from setuptools import setup, find_packages

# %%

# Tutorials
# https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/
# https://python-packaging.readthedocs.io/en/latest/non-code-files.html
# https://choosealicense.com/
# https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/

# %% read the contents of your README file

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# %%

setup(
    name="tensor_routines",
    version="0.1.0",
    description="Routines in numpy and sympy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mauricio Fernández",
    author_email="mauricio.fernandez.lb@gmail.com",
    url="https://github.com/mauricio-fernandez-l/tensor_routines",
    license="MIT",
    python_requires=">=3.9, <3.10.*",
    packages=find_packages(where=".", include=["tensor_routines*"]),
    classifiers=[
        "Intended Audience :: End Users",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
    ],
    keywords=["minimal", "example"],
    zip_safe=False,  # no idea how to use this
    install_requires=["numpy", "sympy"],
)
