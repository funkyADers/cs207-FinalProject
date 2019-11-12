import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="funkyAD-funkyADers", 
    version="0.0.1",
    author="Anna Zink, Johannes K. Kolberg, Fabio Pruneri, Tyler Yoo",
    author_email="",
    description="An implementation of Auto Differentiaion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/funkyADers/cs207-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)