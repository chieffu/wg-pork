# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='poker-game',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'opencv-python',
        'Pillow',
        'numpy',
        'mss',
        'pyautogui',
        'keyboard'
    ],
    entry_points={
        'console_scripts': [
            'wg-pork=game_controller:main',
        ],
    },
    author='你的名字',
    author_email='你的邮箱',
    description='A game controller for a specific game',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/你的用户名/wg-pork',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
