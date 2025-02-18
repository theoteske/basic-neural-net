from setuptools import setup, find_packages

setup(
    name="basic-neural-net",  # Replace with your preferred package name
    version="0.1.0",
    packages=find_packages(where="basic_neural_net"),
    package_dir={"": "basic_neural_net"},
    install_requires=[
        "numpy",
        "scipy",
    ],
    extras_require={
        "examples": [
            "tensorflow",  # For MNIST dataset
            "matplotlib",  # For visualization
            "jupyter",     # For running notebooks
        ],
    },
    python_requires=">=3.7",
    author="Theo Teske",
    description="A NumPy-based neural network implementation built from scratch.",
    license="MIT", 
)