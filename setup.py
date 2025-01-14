from setuptools import setup

setup(
    name='hallucination_gen',
    version='0.1',
    description='A tool for generating hallucinated datasets using transformation strategies and LLMs.',
    url='',
    packages=['hallucination_gen'],
    license='',
    author='Milan Nyist',
    python_requires='',
    install_requires=[
        'pandas~=2.2.2',
        'spacy~=3.5.0',
        'click~=8.1.7',
        'datasets~=2.19.0',
        'requests~=2.31.0',
        'pyyaml~=6.0.1',
        'numpy~=1.26.4',
        'aiohttp~=3.9.5',
        'tqdm~=4.66.2',
        'huggingface-hub~=0.26.2',
    ]
)
