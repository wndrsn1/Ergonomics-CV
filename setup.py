from setuptools import setup, find_packages

setup(
    name='ergo_cv',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pyqt5',
        'opencv-python',
        'mediapipe'
    ],
    entry_points={
        'console_scripts': [
            # Add any console scripts here
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)