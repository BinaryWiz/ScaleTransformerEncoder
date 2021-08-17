from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='Scale Transformer Encoder',
   version='0.1',
   description='A Transformer encoder where the embedding size (dmodel) can be changed.',
   license="MIT",
   long_description=long_description,
   author='Jason Acheampong',
   author_email='jason.acheampong24@gmail.com',
   url="https://github.com/Mascerade/ScaleTransformerEncoder",
   packages=['scale_transformer_encoder'],
   install_requires=['pytorch']
)