from setuptools import setup

setup(
    name='rp',
    version='0.01',
    description='Reciprocal Perspective',
    url='https://github.com/GreenCUBIC/RP',
    author='Daniel Kyrollos',
    author_email='daniel.g.kyrollos@gmail.com',
    license='MIT',
    packages=['rp'],
    install_requires=[
        'tqdm',
        'kneed',
        'pandas>=1.0.3',
        'numpy>=1.18.1',
        'matplotlib',
        'seaborn'
    ],
    zip_safe=False
)
