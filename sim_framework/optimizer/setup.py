from setuptools import setup, find_packages

setup(
    name="systolic_optimizer",
    version="0.1.0",
    description="TensorFlow to Systolic Array Optimization Framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.12.0",
        "numpy>=1.23.0",
    ],
    extras_require={
        "hardware": ["cocotb>=1.8.0", "cocotb-bus>=0.2.1"],
        "viz": ["matplotlib>=3.7.0", "pandas>=2.0.0", "tqdm>=4.65.0"],
        "dev": ["pytest>=7.3.0", "pytest-cov>=4.1.0", "black>=23.3.0", "flake8>=6.0.0"],
    },
)
