import setuptools
import subprocess


description = (
    "An easy to use framework inspired by Tensorflow's keras but for PyTorch users."
)
long_description = "An easy to use framework inspired by Tensorflow's keras but for PyTorch users."

library_version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

setuptools.setup(
    name="torch-keras",
    version=library_version,
    author="Bipin Krishnan P",
    author_email="bipinkrishnan1999.bk@gmail.com",
    license="MIT",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bipinkrishnan/torchkeras",
    packages=setuptools.find_packages(),
    keywords=[],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[],
)