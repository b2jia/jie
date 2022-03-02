from setuptools import setup

setup(
    name='jie',
    version='0.1.1',
    python_requires='>=3.8',
    author='Bojing (Blair) Jia',
    packages=['jie'],
    install_requires=['numpy>=1.20.3',
                      'pandas>=1.3.4',
                      'matplotlib>=3.4.3',
                      'seaborn>=0.11.2',
                      'scipy>=1.7.2',
                      'tqdm>=4.62.3',
                      'scikit-learn>=1.0.1',
                      'python-igraph>=0.9.6'])