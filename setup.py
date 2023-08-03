from setuptools import setup, find_packages



setup(
    name='CUDG',
    version='0.0.2',
    description='CUDG is for gpuarray type for convenience',
    author='Daegeun02',
    author_email='redhawkdg02@gmail.com',
    url='https://github.com/Daegeun02/cudg.git',
    install_requires=[],
    packages=find_packages( exclude=[] ),
    keywords=['CUDG', 'by daegeun', 'for convenience'],
    python_requires='>=3.7',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)