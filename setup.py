import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyramids-AEljarrat", # Replace with your own username
    version="0.0.1",
    author="Alberto Eljarrat",
    author_email="alberto.eljarrat@gmail.com",
    description="Scripts for SEM segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AEljarrat/pyramids",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'matplotlib',
        'imageio',
        'scipy',
        'scikit-image', ],
)