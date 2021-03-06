# Copyright 2018-2019 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Utilities to streamline GPyOpt interfaces for ML'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import shutil
import warnings

from collections import OrderedDict, namedtuple
from csv import DictReader, DictWriter
from tempfile import NamedTemporaryFile
from contextlib import contextmanager

import GPy
import GPyOpt
import numpy as np
import param

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"
__all__ = [
    'GPyOptObjectiveWrapper',
    'BayesianOptimizationParams',
    'bayesopt',
]


class GPyOptObjectiveWrapper(object):
    '''Wrap an objective function to be compatible with GPyOpt

    This wrapper uses inspection to determine the names of the variables
    of the wrapped function.

    Parameters
    ----------
    f : function
        The objective function to be wrapped

    Properties
    ----------
    f : function
    param_dict : OrderedDict
        A dictionary whose keys are argument names and whose values are
        named tuples of ('type', 'domain', 'value')
    num_positional : int
        The number of positional arguments in `f`
    num_keyword : int
        The number of keyword arguments in `f`
    can_add_parameter : bool
        Whether `f` can be called with additional keyword arguments
    '''

    _Variable = namedtuple('_Variable', ['type', 'domain', 'value'])

    def __init__(self, f):
        self.f = f
        arg_names, _, kwargs_name, defaults = inspect.getargspec(f)
        self.can_add_parameter = kwargs_name is not None
        if defaults is None:
            defaults = tuple()
        self.num_keyword = len(defaults)
        self.num_positional = len(arg_names) - self.num_keyword
        self.param_dict = OrderedDict()
        for arg_name in arg_names[:self.num_positional]:
            self.param_dict[arg_name] = None
        for arg_name, default in zip(
                arg_names[self.num_positional:], defaults):
            self.param_dict[arg_name] = self._Variable(
                'fixed', None, default)

    def add_parameter(self, arg_name):
        if not self.can_add_parameter:
            raise ValueError(
                'objective function does not have dynamic keyword arguments')
        self.param_dict.setdefault(arg_name, None)
        self.num_keyword += 1

    def get_fixed_names(self):
        '''set : the names of any fixed parameters (already set)'''
        return set(
            k
            for (k, v) in self.param_dict.items()
            if v is not None and v.type == 'fixed'
        )

    def get_variable_names(self):
        '''set : the names of any variable parameters (already set)'''
        return set(
            k
            for (k, v) in self.param_dict.items()
            if v is not None and v.type != 'fixed'
        )

    def get_unset(self):
        '''set : names of parameters that have yet to be specified'''
        return set(k for (k, v) in self.param_dict.items() if v is None)

    def set_fixed_parameter(self, arg_name, value):
        if isinstance(arg_name, int):
            arg_name = tuple(self.param_dict.keys())[arg_name]
        self.param_dict[arg_name] = self._Variable('fixed', None, value)

    def set_variable_parameter(self, arg_name, gpyopt_type, domain):
        if gpyopt_type not in {'continuous', 'discrete', 'categorical'}:
            raise ValueError(
                'gpyopt_type must be continuous, discrete, or categorical')
        if isinstance(arg_name, int):
            arg_name = tuple(self.param_dict.keys())[arg_name]
        self.param_dict[arg_name] = self._Variable(gpyopt_type, domain, None)

    def _raise_if_unset(self):
        unset = self.get_unset()
        if unset:
            raise ValueError(
                'The following params must be set first: {}'.format(unset))

    def get_domain(self):
        '''Return box constraints for GPyOpt'''
        self._raise_if_unset()
        d = []
        for arg_name, param in self.param_dict.items():
            if param.type == 'fixed':
                continue
            if param.type == 'categorical':
                domain = np.arange(len(param.domain))
            else:
                domain = np.array(param.domain, dtype=np.float64)
            d.append({
                'name': arg_name,
                'type': param.type,
                'domain': domain,
                'dimensionality': 1,
            })
        return d

    def _kwargs2serial(self, kwargs):
        for name, val in kwargs.items():
            # for some reason, bool
            if not isinstance(val, (str, float, int, np.float, np.int, bool)):
                param = self.param_dict[name]
                if param.type == 'fixed':
                    # don't care about re-reading it
                    kwargs[name] = str(val)
                    continue
                # we can only be non-string, categorical here. Easiest way to
                # deal with this is to use the index, and prepend the string
                # with pounds until it no longer matches any entry in the
                # domain. Sure, it's hard to read, but that's your fault for
                # using a weird categorical variable.
                idx = -1
                for idx, pval in enumerate(param.domain):
                    if pval == val:
                        break
                val = str(idx)
                matches_another_val = True
                while matches_another_val:
                    val = '#' + val
                    matches_another_val = False
                    for pval in param.domain:
                        if pval == val:
                            matches_another_val = True
                            break
            kwargs[name] = str(val)

    def _serial2kwargs(self, kwargs):
        for name, val in kwargs.items():
            param = self.param_dict.get(name, None)
            if param is None or param.type == 'fixed':
                continue
            if param.type in {'continuous', 'discrete'}:
                kwargs[name] = type(param.domain[0])(val)
                continue
            found_exact_val = False
            for pval in param.domain:
                if (
                        (pval is True and val == 'True') or
                        (pval is False and val == 'False')):
                    kwargs[name] = pval
                    found_exact_val = True
                    break
                try:
                    # floats, ints, strings
                    if (val == pval) or np.isclose(float(val), pval):
                        kwargs[name] = pval
                        found_exact_val = True
                        break
                except (ValueError, TypeError):
                    pass
            if found_exact_val:
                continue
            # we've padded an index with #s
            val = int(val.lstrip('#'))
            kwargs[name] = param.domain[val]

    def x2kwargs(self, x):
        '''Convert GPyOpt function input to objective function kwargs'''
        self._raise_if_unset()
        i = 0
        d = dict()
        x = x.flatten()
        for arg_name, param in self.param_dict.items():
            if param.type == 'fixed':
                d[arg_name] = param.value
            elif param.type == 'categorical':
                d[arg_name] = param.domain[int(x[i])]
                i += 1
            elif param.type == 'discrete':
                # this ensures we use the original type of the argument
                d[arg_name] = type(param.domain[0])(x[i])
                i += 1
            else:
                d[arg_name] = x[i]
                i += 1
        if i != len(x):
            raise ValueError(
                'mismatch between GPyOpt sample and wrapper kwargs')
        return d

    def kwargs2x(self, kwargs):
        '''Convert objective function kwargs to GPyOpt function input'''
        self._raise_if_unset()
        x = []
        for arg_name, param in self.param_dict.items():
            if param.type == 'categorical':
                x.append(float(param.domain.index(kwargs[arg_name])))
            elif param.type in {'continuous', 'discrete'}:
                x.append(float(kwargs[arg_name]))
        return np.array(x)

    def read_history_to_X_Y(self, fp, objective_entry='Y'):
        '''Read history file (csv) into initial inputs and outputs for GPyOpt

        Parameters
        ----------
        fp : str or file_object
            A file pointer or path to the history file
        objective_entry : str, optional
            The column under which the objective's values are stored

        Returns
        -------
        X, Y : array_like, array_like
            Arguments `X` and `Y` when constructing a GPyOpt.method object
        '''
        self._raise_if_unset()
        if isinstance(fp, str):
            fp = open(fp)
            close = True
        else:
            close = False
        try:
            reader = DictReader(fp)
            X = []
            Y = []
            for row in reader:
                self._serial2kwargs(row)
                X.append(self.kwargs2x(row))
                Y.append(float(row[objective_entry]))
            return np.vstack(X), np.array(Y)[..., np.newaxis]
        finally:
            if close:
                fp.close()

    def write_X_Y_to_history(
            self, fp, X, Y, objective_entry='Y', write_fixed=False):
        '''Write a history file (csv) from inputs and outputs of GPyOpt

        Parameters
        ----------
        fp : str or file_object
            A file pointer or path to the history file
        X : array_like
            The GPyOpt.method's `X` attribute (inputs)
        Y : array_like
            The GPyOpt.method's `Y` attribute (outputs)
        objective_entry : str, optional
            The column under which the objective's values should be stored
        write_fixed : bool, optional
            Whether the fixed parameters should be written to file
        '''
        self._raise_if_unset()
        if isinstance(fp, str):
            fp = open(fp, 'w')
            close = True
        else:
            close = False
        try:
            if write_fixed:
                fieldnames = list(self.param_dict.keys())
                extra_entries = dict(
                    (k, v) for (k, v) in self.param_dict.items()
                    if v.type == 'fixed'
                )
            else:
                fieldnames = [
                    k for (k, v) in self.param_dict.items()
                    if v.type != 'fixed'
                ]
                extra_entries = dict()
            fieldnames.insert(0, objective_entry)
            writer = DictWriter(fp, fieldnames, extrasaction='ignore')
            writer.writeheader()
            for x, y in zip(X, Y):
                kwargs = self.x2kwargs(x)
                kwargs.update(extra_entries)
                kwargs[objective_entry] = float(y)
                self._kwargs2serial(kwargs)
                writer.writerow(kwargs)
        finally:
            if close:
                fp.close()

    def get_gpyopt_func(self):
        '''Returns a function that GPyOpt can optimize'''
        self._raise_if_unset()

        def _f(x):
            kwargs = self.x2kwargs(x)
            return self.f(**kwargs)
        return _f


class BayesianOptimizationParams(param.Parameterized):
    kernel_type = param.ObjectSelector(
        'matern52', objects=['matern52', 'matern32', 'rbf', 'mlp'],
        doc='The type of kernel used in a Gaussian Process optimization'
    )
    initial_design_samples = param.Integer(
        5, bounds=(0, None),
        doc='The number of initial samples before optimization. History counts'
        'towards this'
    )
    initial_design_name = param.ObjectSelector(
        'random', objects=['random', 'grid'],
        doc='How to sample initial samples before optimization'
    )
    model_type = param.ObjectSelector(
        'gp_mcmc', objects=['gp', 'gp_mcmc'],
        doc='How to model the surrogate objective function. gp: gaussian '
        'process with fixed kernel parameters. gp_mcmc : gaussian process '
        'with hybrid monte carlo estimates of parameters'
    )
    acquisition_function = param.ObjectSelector(
        'ei', objects=['ei', 'mpi', 'lcb'],
        doc='How to choose the next sample given previous samples and results.'
        ' ei: expected improvement. mpi: minimum probability of improvement. '
        ' lcb: lower confidence bound'
    )
    acquisition_optimizer = param.ObjectSelector(
        'lbfgs', objects=['lbfgs', 'DIRECT', 'cma'],
        doc='How to minimize the acquisition function. lbfgs: L-BFGS. DIRECT: '
        'Dividing Rectangles. CMA: Covariance Matrix Adaptation'
    )
    noise_var = param.Number(
        0, bounds=(0, None), inclusive_bounds=(True, False),
        doc='The variance created by noise in the objective function'
    )
    max_samples = param.Integer(
        None, bounds=(0, None),
        doc='If set, the max number of samples. history and '
        'initial_design_samples both count towards this. E.g. max_samples=5, '
        'initial_design_numdata=4, and 3 samples have been loaded from a '
        'history file. 1 sample will be taken to complete the initial design, '
        ' 1 according to the acquisition function over the initial design\'s '
        'samples'
    )
    log10_min_diff = param.Integer(
        -8, bounds=(None, 0), inclusive_bounds=(False, False),
        doc='The minimum Euclidean difference between consecutive samples that'
        ' needs to be maintained to keep running the optimization. Note that '
        'this is over the domain, not the range'
    )
    seed = param.Integer(
        None,
        doc='The seed to set numpy\'s global random state to. If set, after '
        'every new sample, the global seed will be set to seed + sample_num. '
        'If unset, numpy\'s random state will not be touched'
    )
    log_after_iters = param.Integer(
        None, bounds=(1, None),
        doc='If set, log to the history file (if available) after this many '
        'iterations, plus the final iteration'
    )
    jitter = param.Number(
        0.01, bounds=(0, None), inclusive_bounds=(False, False),
        doc='larger values mean more exploration'
    )
    write_fixed = param.Boolean(
        False,
        doc='Whether to write fixed parameters to history file'
    )


class _DSWrapper(GPyOpt.Design_space):
    '''Create a design space for a GPyOptObjectiveWrapper

    Parameters
    ----------
    wrapped : GPyOptObjectiveWrapper
        The wrapped function to make a design space for.
        ``wrapped.get_unset()`` must be empty
    constraints : sequence, optional
        A sequence of functions whose only parameters match those of the
        wrapped function. The function do not need to list all parameters, but
        cannot include any parameters with names not listed in the wrapped
        function. Given a sample, a constraint is expected to return ``True``
        if the sample adheres to the restraint, ``False`` otherwise.
    '''

    def __init__(self, wrapper, constraints=None):
        if len(wrapper.get_unset()):
            raise ValueError('All parameters must be set in objective')
        self.wrapper = wrapper
        if constraints is not None:
            all_param_names = set(wrapper.param_dict)
            new_constraints = []
            self.constraint_args = []
            for constraint in constraints:
                try:
                    name = constraint['name']
                    constraint = constraint['constraint']
                except TypeError:
                    name = 'constraint_{}'.format(len(new_constraints) + 1)
                arg_names, _, kwargs_name, defaults = inspect.getargspec(
                    constraint)
                num_kwargs = len(defaults) if defaults else 0
                pos_names = set(arg_names[:len(arg_names) - num_kwargs])
                missing_params = pos_names - all_param_names
                if kwargs_name is not None:
                    # pass everything to the constraint
                    constraint_kwargs = None
                else:
                    constraint_kwargs = arg_names
                if missing_params:
                    raise ValueError(
                        'constraint contains parameters {} which are not '
                        'arguments to the objective'.format(missing_params))
                new_constraints.append({
                    'name': name,
                    'constraint': constraint,
                    'kwargs': constraint_kwargs
                })
            constraints = new_constraints
        super(_DSWrapper, self).__init__(
            wrapper.get_domain(), constraints=constraints)

    def indicator_constraints(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] != self.objective_dimensionality:
            X = self.zip_inputs(X)
        Ix = np.ones((X.shape[0], 1))
        if self.constraints is not None:
            for constraint in self.constraints:
                constraint_kwargs = constraint['kwargs']
                constraint = constraint['constraint']
                for i, x in enumerate(X):
                    if Ix[i, 0]:
                        kwargs = self.wrapper.x2kwargs(x)
                        if constraint_kwargs is not None:
                            kwargs = {
                                x: kwargs[x] for x in constraint_kwargs
                                if x in kwargs
                            }
                        if not constraint(**kwargs):
                            Ix[i, 0] = 0.
        return Ix


@contextmanager
def _inject_as_design_space(ds):
    old = GPyOpt.optimization.anchor_points_generator.Design_space

    class _Injected(object):
        def __new__(cls, space, constraints=None, store_noncontinuous=False):
            # make sure this injection is safe. If any of these asserts fail,
            # then this hack is broken
            for s, s2 in zip(space, ds.config_space):
                assert '_'.join(s['name'].split('_')[:-1]) == s2['name']
                assert s2['type'] == s['type']
                assert np.allclose(s2['domain'], s['domain'])
                assert s2['dimensionality'] == s['dimensionality']
            assert constraints == ds.constraints
            assert store_noncontinuous == ds.store_noncontinuous
            return ds
    try:
        GPyOpt.optimization.anchor_points_generator.Design_space = _Injected
        yield
    finally:
        GPyOpt.optimization.anchor_points_generator.Design_space = old


def bayesopt(wrapper, params, history_file=None, constraints=None):
    '''Perform bayesian optimization on a wrapped objective function

    Parameters
    ----------
    wrapper : GPyOptObjectiveWrapper
        The wrapped objective function. All parameters should be set by
        this point
    params : BayesianOptimizationParams
    history_file : str, optional
        A path to a history file, where past samples are saved and loaded.
        Floating points, integers, and strings will be serialized as expected.
        Categorical variables that are not strings will be serialized with a
        pound ('#') followed by their index in the domain
    constraints : sequence, optional
        A sequence of functions whose only parameters match those of the
        wrapped function. The function do not need to list all parameters, but
        cannot include any parameters with names not listed in the wrapped
        function. Given a sample, a constraint is expected to return ``True``
        if the sample adheres to the restraint, ``False`` otherwise.

    Returns
    -------
    best : dict
        A dictionary of key-value pairs of the settings that lead to the best
        objective value so far
    '''
    if len(wrapper.get_unset()):
        raise ValueError('All parameters must be set in objective')
    objective = GPyOpt.core.task.SingleObjective(wrapper.get_gpyopt_func())
    space = _DSWrapper(wrapper, constraints=constraints)
    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(
        space, optimizer=params.acquisition_optimizer)
    input_dim = len(wrapper.get_variable_names())
    if params.kernel_type == 'matern52':
        kernel = GPy.kern.Matern52(input_dim)
    elif params.kernel_type == 'matern32':
        kernel = GPy.kern.Matern32(input_dim)
    elif params.kernel_type == 'rbf':
        kernel = GPy.kern.RBF(input_dim)
    else:
        kernel = GPy.kern.MLP(input_dim)
    if params.model_type == 'gp':
        model_class = GPyOpt.models.GPModel
        if params.acquisition_function == 'ei':
            acquisition_class = GPyOpt.acquisitions.AcquisitionEI
        elif params.acquisition_function == 'mpi':
            acquisition_class = GPyOpt.acquisitions.AcquisitionMPI
        else:
            acquisition_class = GPyOpt.acquisitions.AcquisitionLCB
    else:
        model_class = GPyOpt.models.GPModel_MCMC
        if params.acquisition_function == 'ei':
            acquisition_class = GPyOpt.acquisitions.AcquisitionEI_MCMC
        elif params.acquisition_function == 'mpi':
            acquisition_class = GPyOpt.acquisitions.AcquisitionMPI_MCMC
        else:
            acquisition_class = GPyOpt.acquisitions.AcquisitionLCB_MCMC
    if params.noise_var == 0:
        noise_var = None
        exact_feval = True
    else:
        noise_var = params.noise_var
        exact_feval = False
    model = model_class(
        kernel=kernel, noise_var=noise_var, exact_feval=exact_feval)
    acquisition = acquisition_class(
        model, space, acquisition_optimizer, jitter=params.jitter)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    log_after_iters = float('inf')
    max_samples = params.max_samples
    if max_samples is None:
        max_samples = float('inf')
    X = Y = np.empty(0)
    if history_file is not None:
        if params.log_after_iters:
            log_after_iters = params.log_after_iters
        try:
            X, Y = wrapper.read_history_to_X_Y(history_file)
        except IOError:
            pass
    rem = max(0, max_samples - len(X))
    initial_design_samples = max(params.initial_design_samples - len(X), 0)
    initial_design_samples = min(rem, initial_design_samples)

    def write_to_history(X, Y):
        if history_file:
            tmp_file = NamedTemporaryFile(mode='w', delete=False)
            wrapper.write_X_Y_to_history(
                tmp_file, X, Y, write_fixed=params.write_fixed)
            tmp_file.close()
            shutil.move(tmp_file.name, history_file)

    if params.seed is not None:
        np.random.seed(params.seed)
    X_init = GPyOpt.experiment_design.initial_design(
        params.initial_design_name, space,
        params.initial_design_samples)
    if (
            len(X) and
            len(X_init) and
            not np.allclose(X_init[:len(X)], X[:len(X_init)])):
        if params.initial_design_name == 'grid':
            raise ValueError(
                'Points were to be initially sampled from a grid, but the '
                'history file had different initial samples'
            )
        elif params.seed is not None:
            warnings.warn(
                'history file does not share initial samples as those '
                'that were just generated. Continuing will make it difficult '
                'to reproduce results'
            )
    X_init = X_init[len(X):max_samples if max_samples < float('inf') else None]
    assert len(X_init) == initial_design_samples
    samples_before_log = log_after_iters
    while initial_design_samples:
        cur_num = min(initial_design_samples, samples_before_log)
        X_cur, X_init = X_init[:cur_num], X_init[cur_num:]
        bo = GPyOpt.methods.ModularBayesianOptimization(
            model, space, objective, acquisition, evaluator, X_cur)
        bo.run_optimization(0)
        X = np.vstack([X.reshape(-1, bo.X.shape[1]), bo.X])
        Y = np.vstack([Y.reshape(-1, bo.Y.shape[1]), bo.Y])
        initial_design_samples -= cur_num
        rem -= cur_num
        samples_before_log -= cur_num
        if not samples_before_log:
            samples_before_log = log_after_iters
            write_to_history(X, Y)
    assert len(X) >= min(max_samples, params.initial_design_samples)
    # As of writing, GPyOpt only updates model parameters once - on
    # initialization of 'bo'. Those parameters are based on X and Y. To
    # ensure that restarting the process gets the same model parameters, we
    # pretend as though we have only initial_design_samples, then update X and
    # Y later
    bo = GPyOpt.methods.ModularBayesianOptimization(
        model, space, objective, acquisition, evaluator,
        X, Y,
        normalize_Y=False,
    )
    while rem:
        if params.seed is not None:
            np.random.seed(params.seed)
            cur_num = 1
        else:
            cur_num = min(rem, samples_before_log)
        with _inject_as_design_space(space):
            bo.run_optimization(cur_num, eps=10 ** params.log10_min_diff)
        X_new, Y_new = bo.get_evaluations()
        assert np.allclose(X_new[:len(X)], X)
        if len(X_new) - len(X) < cur_num:
            # we were less than the minimum difference
            X, Y = X_new, Y_new
            break
        X, Y = X_new, Y_new
        rem -= cur_num
        samples_before_log -= cur_num
        if not samples_before_log:
            samples_before_log = log_after_iters
            write_to_history(X, Y)
    write_to_history(X, Y)
    best = wrapper.x2kwargs(bo.X[np.argmin(bo.Y)])
    for fixed_name in wrapper.get_fixed_names():
        best.pop(fixed_name)
    return best
