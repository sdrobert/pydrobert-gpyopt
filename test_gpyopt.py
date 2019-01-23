from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tempfile import mkdtemp
from shutil import rmtree

import pytest
import pydrobert.gpyopt as gpyopt
import numpy as np

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


@pytest.fixture
def temp_dir():
    dir_name = mkdtemp()
    yield dir_name
    rmtree(dir_name)


def _polynomial(x, d, shift):
    return x ** d + shift


def _squared(x, shift, noise=0.0):
    return _polynomial(x, 2, shift + noise * np.random.randn())


def test_simple_objective():
    wrapper = gpyopt.GPyOptObjectiveWrapper(_squared)
    wrapper.set_fixed_parameter('shift', 1.)
    wrapper.set_variable_parameter('x', 'continuous', (-1., 1.))
    params = gpyopt.BayesianOptimizationParams(
        model_type='gp',
        kernel_type='rbf',
        seed=1,
        max_samples=10,
        noise_var=0,
    )
    best_1 = gpyopt.bayesopt(wrapper, params)
    best_2 = gpyopt.bayesopt(wrapper, params)
    assert best_1['x'] == best_2['x']
    params.log10_min_diff = -3
    params.max_samples = None
    best_3 = gpyopt.bayesopt(wrapper, params)
    # note that log10_min_diff is not the same as the absolute
    # value between best results
    assert abs(_squared(best_3['x'], 1.) - 1.) < 1e-2


def test_multiple_variable_types():
    wrapper = gpyopt.GPyOptObjectiveWrapper(_polynomial)
    wrapper.set_variable_parameter('shift', 'categorical', (10, 0, 1))
    wrapper.set_variable_parameter('x', 'continuous', (.1, .9))
    wrapper.set_variable_parameter('d', 'discrete', (-1, 5))
    params = gpyopt.BayesianOptimizationParams(
        model_type='gp',
        seed=1,
        log10_min_diff=-2,
        initial_design_samples=10,
        noise_var=0,
    )
    best = gpyopt.bayesopt(wrapper, params)
    assert not best['shift']
    assert best['d'] == 5


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_restart_with_seed_gives_same_predictions(temp_dir, seed):
    wrapper = gpyopt.GPyOptObjectiveWrapper(
        lambda **args: _squared(**args, noise=0.1))
    wrapper.set_variable_parameter('x', 'continuous', (-1., 1.))
    wrapper.set_variable_parameter('shift', 'continuous', (10, 11))
    hist_path = os.path.join(temp_dir, 'hist.csv')
    params = gpyopt.BayesianOptimizationParams(
        max_samples=6,
        model_type='gp',
        seed=seed,
        noise_var=0.1,
    )
    gpyopt.bayesopt(wrapper, params, hist_path)
    params.max_samples = 8
    best_1 = gpyopt.bayesopt(wrapper, params, hist_path)
    best_2 = gpyopt.bayesopt(wrapper, params)
    assert abs(best_1['x'] - best_2['x']) < 1e-5


def test_strings_are_provided():

    def _obj(mystring):
        assert isinstance(mystring, str)
        return int(mystring[0])

    wrapper = gpyopt.GPyOptObjectiveWrapper(_obj)
    wrapper.set_variable_parameter('mystring', 'categorical', '1234')
    params = gpyopt.BayesianOptimizationParams(
        initial_design_numdata=2,
        max_samples=10,
    )
    best = gpyopt.bayesopt(wrapper, params)
    assert best['mystring'] == '1'
