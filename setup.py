from setuptools import setup, find_packages

setup(
    name='QuantGrad',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
    'graphviz'
            ],
    author='Surya Narayanaa N T',
    author_email='suryanarayanaant@gmail.com',
    description='A Small(Quant) Auto Grad Library with visualization of individual operations within Neurons',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SuryaNarayanaa/QuantGrad',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)