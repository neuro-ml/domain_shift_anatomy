from pathlib import Path

from setuptools import setup, find_packages


# automatically detect the lib name
dirs = {d.name for d in Path(__file__).resolve().parent.iterdir() if d.is_dir()}
dirs -= {'configs', 'notebooks', 'scripts', 'tests'}
dirs = [n for n in dirs if not n.startswith(('_', '.'))]
dirs = [n for n in dirs if not n.endswith(('.egg-info', ))]

assert len(dirs) == 1, dirs
name = dirs[0]

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

setup(
    name=name,
    packages=find_packages(include=(name,)),
    descriprion='A collection of tools for deep learning experiments',
    install_requires=requirements,
    # OPTIONAL: uncomment if needed
    # python_requires='>=3.6',
)
