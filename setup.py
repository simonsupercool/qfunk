import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qfunk",
    version="0.1.2",
    author="Simon Milz, Joshua Morris",
    author_email="Simon.Milz@oeaw.ac.at, joshua.morris@univie.ac.at",
    description="quantum information library",
    long_description_content_type="text/markdown",
    url="https://github.com/simonsupercool/qfunk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)