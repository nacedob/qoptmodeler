from setuptools import setup, find_packages

setup(
    name="qumodeler",
    version="0.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.2.0",
        "pytest>=7.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9.0",
)
