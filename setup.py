import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="research",
    version="0.0.1",
    author="Truong Tuan Anh",
    author_email="truongtuananhsamson@gmail.com",
    description="Personal research on Masked Autoencoder and Semantic Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages()
)
