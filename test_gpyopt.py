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


def _polynomial(x, d, shift, noise=None):
    if noise is None:
        noise = 0.
    elif noise == 'random':
        noise = np.random.random()
    return x ** d + shift + noise


def _squared(x, shift):
    return _polynomial(x, 2, shift)


def _day_cannot_be_tuesday(day, some_other_value=1):
    return day != 'tuesday'


def _int_is_even(int_):
    return bool(1 - int_ % 2)


def _enforcer_objective(int_, day):
    assert _day_cannot_be_tuesday(day)
    assert _int_is_even(int_)
    return np.random.randn()


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


# def test_constraints():
#     wrapper = gpyopt.GPyOptObjectiveWrapper(_enforcer_objective)
#     wrapper.set_variable_parameter('x', 'discrete' (0, 4))
#     wrapper.set_variable_parameter('day', 'categorical', ('monday', 'tuesday'))
#     params = gpyopt.BayesianOptimizationParams(
#         seed=10
#     )


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


@pytest.mark.parametrize('inter_max_samples', [4, 5, 6])
def test_restart_with_seed_gives_same_predictions(temp_dir, inter_max_samples):
    wrapper = gpyopt.GPyOptObjectiveWrapper(_polynomial)
    wrapper.set_variable_parameter('x', 'continuous', (-1., 1.))
    wrapper.set_fixed_parameter('d', 2)
    wrapper.set_fixed_parameter('shift', 1)
    hist_path = os.path.join(temp_dir, 'hist.csv')
    params = gpyopt.BayesianOptimizationParams(
        max_samples=inter_max_samples,
        model_type='gp',
        seed=1,
    )
    gpyopt.bayesopt(wrapper, params, hist_path)
    params.max_samples = 8
    best_1 = gpyopt.bayesopt(wrapper, params, hist_path)
    best_2 = gpyopt.bayesopt(wrapper, params)
    assert abs(best_1['x'] - best_2['x']) < 1e-5


def test_can_serialize_lots_of_stuff(temp_dir):
    def _obj(**kwargs):
        return 0.

    wrapper = gpyopt.GPyOptObjectiveWrapper(_obj)
    wrapper.set_fixed_parameter('a', object())
    wrapper.set_variable_parameter('b', 'continuous', (0., 1.))
    wrapper.set_variable_parameter('c', 'discrete', (0, 1))
    d_tup = (1, None, 'None', 0., object())
    wrapper.set_variable_parameter('d', 'categorical', d_tup)
    hist_path = os.path.join(temp_dir, 'hist.csv')
    params = gpyopt.BayesianOptimizationParams(
        initial_design_samples=4,
        max_samples=4,
        seed=5,
    )
    best_1 = gpyopt.bayesopt(wrapper, params, hist_path)
    assert best_1['d'] in d_tup
    best_2 = gpyopt.bayesopt(wrapper, params, hist_path)
    assert best_1['c'] == best_2['c']
    assert abs(best_1['b'] - best_2['b']) < 1e-5
    assert best_1['d'] == best_2['d']


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
