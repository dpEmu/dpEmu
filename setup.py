import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dpemu",
    version="0.0.1",
    author="Team dpEmu",
    #author_email="author@example.com",
    description="Tools for emulating data problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dpEmu/dpEmu",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux"
    ],
)
