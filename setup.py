import io
import os
import re

from setuptools import find_packages
from setuptools import setup

import face_eevee

def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


    author, author_email = re.search('(.*)<(.*)>', face_eevee.__author__).groups()

setup(
    name="face_eevee",
    version=face_eevee.__version__,
    url="https://github.com/alexisfcote/face_eevee",
    license='MIT',
    author="Alexis Fortin-Cote",
    author_email="alexisfcote@gmail.com",

    description="Early test of eos lib for the EEVEE plateform",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests',)),

    install_requires=['imutils', 'numpy', 'opencv-python', 'eos-py', 'dlib', 'matplotlib'] ,

    entry_points={
        'console_scripts': [
            'face_eevee = face_eevee.face_eevee:main',
        ]
    },

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
)
