from setuptools import setup, find_packages

setup(
    name="cmpd_attn",
    version="0.1.0",
    author="Anonymous",
    author_email="",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cmpd_attn",  # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",  # Pytorch package
    ],
    python_requires="<3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update license type
        "Operating System :: OS Independent",
    ],
)
