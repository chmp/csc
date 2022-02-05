import pathlib
import re
from setuptools import setup, PEP420PackageFinder

init_source = (
    pathlib.Path(__file__)
    .parent.resolve()
    .joinpath("src", "csc", "__init__.py")
    .read_text()
)

(version,) = [
    m.group(1)
    for line in init_source.splitlines()
    for m in [re.match(r"^__version__ = \"(.*)\"$", line)]
    if m is not None
]

setup(
    name="csc",
    version=version,
    description="Run python scripts cell by cell.",
    long_description=pathlib.Path("Readme.md").read_text(),
    long_description_content_type="text/markdown",
    author="Christopher Prohm",
    url="https://github.com/chmp/csc",
    author_email="mail@cprohm.de",
    license="MIT",
    packages=PEP420PackageFinder.find("src"),
    package_dir={"": "src"},
    install_requires=[],
    tests_require=["pytest"],
    python_requires=">=3",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
