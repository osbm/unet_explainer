from setuptools import setup, find_packages

setup(
    name = "unet_pytorch",
    version = "0.1.0",
    author = "Osman Bayram",
    author_email = "osmanfbayram@gmail.com",
    description = ("An unet pytorch tutorial"),
    license = "MIT",
    install_requires=['torch', 'torchvision', 'monai', 'albumentations', 'matplotlib', 'numpy', 'pandas', 'tqdm', 'scikit-image', 'scikit-learn', 'seaborn', 'SimpleITK'],
    packages=find_packages(exclude=['scripts']),
)
