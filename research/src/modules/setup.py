import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="self_seg",
    version="0.0.1",
    author="Kokoroou",
    author_email="truongtuananhsamson@gmail.com",
    description="Self-supervised learning for semantic segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/Kokoroou/self-supervised-segmentation",
    install_requires=[
        "torch>=1.6.0",
        "torchvision>=0.7.0",
        "numpy>=1.19.1",
        "scikit-image>=0.17.2",
        "tqdm>=4.48.2",
        "tensorboard>=2.3.0",
        "matplotlib>=3.3.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
