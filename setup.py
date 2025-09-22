from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rankfit",
    version="0.1.1",
    author="Emma Chung",
    author_email="hsiaoyin.chung@gmail.com",
    description="Segment-level ranking quality metrics for machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EMMA-CHUNG/rankfit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
    ],
    project_urls={
        "Bug Reports": "https://github.com/EMMA-CHUNG/rankfit/issues",
        "Source": "https://github.com/EMMA-CHUNG/rankfit",
        "Documentation": "https://github.com/EMMA-CHUNG/rankfit#readme",
    },
)
