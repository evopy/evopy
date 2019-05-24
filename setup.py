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
      install_requires=[
            "numpy",
            "nose",
            "pylint",
      ],
      zip_safe=False)
