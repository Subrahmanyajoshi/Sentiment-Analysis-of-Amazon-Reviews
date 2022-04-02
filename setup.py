from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = []

setup(
    name='Amazon Reviews Analysis',
    version='0.1',
    author='Subrahmanya Joshi',
    author_email='subrahmanyajoshi123@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Sentiment analysis of amazon product reviews',
    requires=[]
)
