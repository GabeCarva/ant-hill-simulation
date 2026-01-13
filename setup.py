"""Setup script for ant-hill-simulation package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ant-hill-simulation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A reinforcement learning environment where ant colonies compete using deep learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ant-hill-simulation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "deep": ["torch>=2.0.0", "torchvision>=0.15.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "flake8>=6.0.0"],
    },
    entry_points={
        "console_scripts": [
            "ant-sim-train=training.train_curriculum:main",
            "ant-sim-demo=scripts.demo_game:main",
            "ant-sim-eval=scripts.evaluate_agents:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt"],
    },
)
