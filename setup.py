from setuptools import setup, find_packages
import os

# Read the contents of README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Define package metadata
setup(
    name="bnc_tracker",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Binocular camera-based 3D position tracking system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bnc_tracker",
    
    # Find all packages automatically
    packages=find_packages(include=["bnc_tracker", "bnc_tracker.*"]),
    
    # Python version compatibility
    python_requires=">=3.6",
    
    # Dependencies
    install_requires=[
        "numpy>=1.18.0",
        "opencv-python>=4.2.0",
        "matplotlib>=3.2.0",
        "websockets>=8.1",
        "keyboard>=0.13.5",  # For keyboard input handling
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
    },
    
    # Command-line scripts
    entry_points={
        "console_scripts": [
            "bnc-tracker=bnc_tracker.scripts.bnc_tracker_app:main",
        ],
    },
    
    # Include non-Python files in the package
    package_data={
        "bnc_tracker": ["resources/*.txt"],
    },
    
    # Include the examples directory in the source distribution but not in the installed package
    # This makes examples available in the source distribution but doesn't clutter the installed package
    data_files=[
        ("examples", [os.path.join("examples", f) for f in os.listdir("examples") if f.endswith(".py")]),
    ],
    
    # Project classification for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  # Change as needed
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords for easier discovery
    keywords="computer vision, 3D tracking, stereo vision, camera tracking",
)