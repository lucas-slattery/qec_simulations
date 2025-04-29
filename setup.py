from setuptools import setup, find_packages

setup(
    name="qec",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "cirq",
        "numpy",
        "stimcirq",
        "adaptive-scheduler",
        "pipefunc",
        "sinter",
        "stim",
    ],
    author="Lucas Slattery",
    description="Quantum Error Correction Simulations",
    url="https://github.com/your-repo/qec_simulations",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)