from setuptools import setup

setup(name='evopy',
      version='0.1',
      description='Evolutionary Strategies made simple',
      url='http://github.com/evopy/evopy',
      author='evopy',
      author_email='info@gandreadis.com',
      license='MIT',
      packages=['evopy'],
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      install_requires=[
          "numpy",
          "nose",
          "pylint",
      ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ])
