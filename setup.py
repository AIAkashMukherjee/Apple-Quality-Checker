
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='Apple Quality',
    version='0.0.1',
    author='Akash Mukherjee',
    author_email='akashmukherjee0000@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages(),
)
