from setuptools import setup
import pkg_resources

with open('requirements.txt') as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='MIMo-BabyBench',
    version='0.1',
    url='https://github.com/babybench/BabyBench',
    license='',
    author='Francisco M. López', 
    packages=['mimoEnv', 'mimoVision', 'mimoVestibular', 'mimoProprioception', 'mimoTouch', 'mimoActuation'],
    install_requires=install_requires,
    author_email='fcomlop@gmail.com',
    description='MIMo library for BabyBench'
)
