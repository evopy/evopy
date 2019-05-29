# evopy

[![Build Status](https://travis-ci.com/evopy/evopy.svg?branch=master)](https://travis-ci.com/evopy/evopy)
[![PyPI](https://img.shields.io/pypi/v/evopy.svg)](https://pypi.org/project/evopy/)
[![Docs](https://readthedocs.org/projects/evopy/badge/?version=latest)](http://evopy.readthedocs.io/)

**Evolutionary Strategies made simple!**

Use evopy to easily optimize a vector of floats in Python.

## 🏗 Installation

All you need to use evopy is [Python 3](https://www.python.org/downloads/)! Run this command to fetch evopy from PyPI:

```
pip install evopy
```

Then you can import `EvoPy` like this:

```python
from evopy import EvoPy
```

## ⏩ Usage

### One-Dimensional Example

Let's say we wanted to find the optimum of a parabola, without using exact methods from calculus! With Evopy, this is as easy as writing the following two lines:

```python
evopy = EvoPy(lambda x: pow(x, 2), 1)
best_coordinates = evopy.run()
```

The main ingredient here is the fitness function (the lambda). This can also be a normal function reference, just make sure that it accepts a float or an array of floats and outputs a single float. The other ingredient is the `1` at the end of the first line: This is the dimensionality of the inputs that you expect in your fitness function. `best_coordinates` will contain an array with a single element, which is the best `x` value the algorithm could find in the default number of generations.

### Multi-Dimensional Example

If the previous example seemed too simple to you, we can also look at the optimum of a more complex, two-dimensional function, like the [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function). We don't have to modify much in our previous code snippet to get this to work:

```python
evopy = EvoPy(
    lambda X: 5 + sum([(x**2 - 5 * np.cos(2 * np.pi * x)) for x in X]), 
    2, 
    generations=1000, 
    population_size=100
)
best_coordinates = evopy.run()
```

Compared to the first example, we have interchanged the fitness function for a more complex one, set the dimensionality to `2`, and given the algorithm more time to find an optimum by setting a higher generation and individual count than the default.

### Docs

For more detailed information on evopy's functionality, have a look at [the docs](http://evopy.readthedocs.io/)!

## ⛏ Development

[Clone this repository](https://github.com/evopy/evopy) and fetch all dependencies from within the cloned directory:

```
pip install .
```

Run all tests with:

```
nosetests
```

To check your code style, run:

```
pylint evopy
```

To measure your code coverage, run:

```
nosetests --with-coverage --cover-package=evopy --cover-html --cover-branches --cover-erase
```
