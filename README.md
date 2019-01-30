# pydrobert-gpyopt
Utilities to streamline GPyOpt interfaces for ML

## How to use
GPyOpt is incredibly powerful, but a tad clunky. This lightweight package
provides two utilities in ``pydrobert.gpyopt`` to make things easier. The
first, ``GPyOptObjectiveWrapper``, wraps a function for use in GPyOpt. The
second, ``bayesopt``, takes a wrapper instance and a
``BayesianOptimizationParams`` instance and handles the optimization loop.
Here's an example:

``` python
import pydrobert.gpyopt as gpyopt
def foo(a, d, b, **kwargs):
    r = a ** d + b
    weirdness = kwargs['weirdness']
    if weirdness == 'flip':
        r *= -1
    elif weirdness == 'null':
        r = 0
    return r
wrapper = gpyopt.GPyOptObjectiveWrapper(foo)
wrapper.set_fixed_parameter('b', 1.)  # 'b' will always be 1
wrapper.set_variable_parameter('a', 'continuous', (-1., 1.))  # a is real
                                                              # btw [-1,1] inc
wrapper.set_variable_parameter('d', 'discrete', (0, 3))  # d is an int
                                                         # btw [0, 3] inc
wrapper.add_parameter('weirdness')  # we can add new parameters as dynamic
                                    # keyword args if the method has a **
                                    # parameter
wrapper.set_variable_parameter(  # weirness one of the elements in the list
    'weirdness', 'categorical', ('flip', 'null', None))
params = gpyopt.BayesianOptimizationParams(
  seed=1,  # setting this makes the bayesian optimization deterministic
           # (assuming foo is deterministic)
  log_after_iters=5,
)
best = gpyopt.bayesopt(wrapper, params, 'hist.csv', constraints=[
    lambda a, d: abs(a - d) > .5,  # constrain the difference between a and
                                   # d to be greater than .5
])
```

If you provide a history file to read/write from, optimization can be
resumed after unexpected interrupts. There are a lot of options to ``bayesopt``
that are listed in ``BayesianOptimizationParams``.

## Installation

GPyOpt currently does not have a Conda build, so pydrobert-gpyopt is available
via PyPI and source install.

``` bash
pip install pydrobert-gpyopt
```
