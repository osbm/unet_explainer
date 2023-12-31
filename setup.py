from setuptools import setup, find_packages

setup(
    name = "unet_pytorch",
    version = "0.3.0",
    author = "Osman Bayram",
    author_email = "osmanfbayram@gmail.com",
    description = ("An unet pytorch tutorial"),
    license = "MIT",
    install_requires=['torch', 'monai', 'albumentations', 'matplotlib', 'numpy', 'pandas', 'tqdm', 'seaborn', 'torchmetrics', "notebook"],
    packages=find_packages(exclude=['scripts']),
)
