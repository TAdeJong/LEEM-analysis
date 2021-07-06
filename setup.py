import setuptools
import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="LEEM-analysis-registration",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Tobias A. de Jong",
    author_email="tobiasadejong@gmail.com",
    description="A package for image registration and LEEM data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TAdeJong/LEEM-analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)
