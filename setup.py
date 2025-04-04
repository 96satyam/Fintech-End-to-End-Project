from setuptools import setup, find_packages 
from typing import List


def get_requirements(file_path:str) -> list[str]:
    """
    This function returns a list of requirements from the given file path.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements



setup(
    name = "fintechproject",
    version = "0.1.0",
    author = "Satyam Tiwari",
    author_email = "satyam.tiwari.9695@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt'),
    description= "Fintech Project",
)