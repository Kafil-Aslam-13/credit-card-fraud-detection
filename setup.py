from setuptools import setup, find_packages

def get_requirements(file_path:str):
    """ Reads Requirement.txt and returns a clean list of dependencies"""
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.strip() for req in requirements]
        requirements=[req for req in requirements if req and not req.startswith("#")]
        if "-e ." in requirements:
            requirements.remove("-e .")
        return requirements

setup(
    name="credit_card_fraud_detection",
    version="0.0.1",
    packages=find_packages(),
    author="Kafil Aslam",
    description="Production Grade end to end Credit Card fraud detection system",
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.11"
)