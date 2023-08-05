from setuptools import setup, find_packages

setup(
   name='effective_sequence_compression',
   version='1.0',
   description='An utils for experiments',
   author='Konstantin Vedernikov',
   author_email='example@mail.ru',
   packages=['effective_sequence_compression.models',
             'effective_sequence_compression.datasets',
             'effective_sequence_compression.training',
             'effective_sequence_compression.metrics',
             'effective_sequence_compression.generator',
             'effective_sequence_compression'],
)
