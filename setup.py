from setuptools import setup

with open('./requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ecg_byte',
    version='1.0',
    packages=['ecg_byte', 'ecg_byte.models', 'ecg_byte.runners',
              'ecg_byte.analysis', 'ecg_byte.preprocess', 'ecg_byte.utils'],
    url='https://github.com/willxxy/ECG-Byte',
    license='MIT',
    author='William Jongwon Han',
    author_email='wjhan@andrew.cmu.edu',
    description='Open source code of ECG-Byte',
    install_requires=required,
)