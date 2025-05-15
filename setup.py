from setuptools import setup, find_packages

setup(
    name="qoptmodeler",
    version="0.3.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.2.0",
        "pytest>=7.0.0",
        "pennylane>=0.40.0",
        "jax>=0.4.0",
        "scipy>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9.0",
)
