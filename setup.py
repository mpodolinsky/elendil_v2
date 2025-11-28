"""
Setup script for ELENDIL v2 package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="elendil-v2",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A multi-agent reinforcement learning environment for UAVs, UGVs, and targets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/elendil_v2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.29.0",
        "pettingzoo>=1.24.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)

