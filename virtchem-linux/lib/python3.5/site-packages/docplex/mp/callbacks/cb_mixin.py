#!/usr/bin/python
# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017
# ---------------------------------------------------------------------------
#
# gendoc: ignore
from docplex.mp.compat23 import izip
from docplex.mp.cplex_engine import CplexEngine


class ModelCallbackMixin(object):
    """
    This mixin class is intended as a bridge between DOcplex expression and constraints
    and CPLEX callback API.
    It is not intended to be instantiated directly, but to be inherited from in custom callbacks
    , jointly with a CPLEX callback type.

    For example, to define a custom BranchCallback in Docplex, define a new class which
    inherits both from ModelCallbackMixin and the legacy callback class BranchCallback.

    Note:
        - `ModelCallbackMixin` should be first in inheritance order,
        - the constructor of the custom callback class must take an `env` parameter to comply
            with the CPLEX API
        - the constructor of the custom callback must call two __init__() methods:
            - one for the cplex callback class, taking the `env` parameter
            - one for the mixin class

    Example:

        class MyBranch(ModelCallbackMixin, cplex.callbacks.BranchCallback):

            def __init__(self, env):
                cplex.callbacks.BranchCallback.__init__(self, env)
                ModelCallbackMixin.__init__(self)

    A custom callback must be registered with a `Model` class using Model.register_callback; this method
    assumes the custom callback has a model setter property to connect the model to the callback.


        See Also:
            :func:`docplex.mp.model.Model.register_callback`


    """

    def __init__(self):
        self._model = None

    @property
    def model(self):
        """ This property is used to get or set the model associated with the mixin.

        An exception is raised if no model has been associated with the mixin.

        :return: an instance of `docplex.mp.Model`
        """
        if not self._model:
            raise ValueError('No model has been attached to the callback.')  # pragma: no cover
        return self._model

    @model.setter
    def model(self, mdl):
        self._model = mdl

    def index_to_var(self, var_index):
        """ This method converts a variable index to a Var object.

        A model must have been associated withthe mixin, otherwise an error is raised.

        :param var_index: A valid variable index, that is a positive integer.

        :return: A Docplex variable with this index, or None.
        """
        assert var_index >= 0
        dv = self.model.get_var_by_index(var_index)
        return dv

    @staticmethod
    def linear_ct_to_cplex(linear_ct):
        cpx_lhs = CplexEngine.linear_ct_to_cplex(linear_ct=linear_ct)
        cpx_rhs = linear_ct.cplex_num_rhs()
        cpx_sense = linear_ct.sense.cplex_code
        return cpx_lhs, cpx_sense, cpx_rhs

    def make_solution_from_vars(self, dvars):
        # build a solution object from array of solution values
        # noinspection PyUnresolvedReferences
        m = self.model
        if dvars:
            indices = [v._index for v in dvars]
            # this calls the Cplex callback method get_values, which crashes if called with empty list
            var_values = super(ModelCallbackMixin, self).get_values(indices)
            var_value_dict = {v: val for v, val in izip(dvars, var_values)}
        else:
            var_value_dict = {}
        return m.new_solution(var_value_dict)


class ConstraintCallbackMixin(ModelCallbackMixin):

    def __init__(self):
        ModelCallbackMixin.__init__(self)
        self.vars = None
        self.cts = []

    def register_constraints(self, cts):
        self.cts.extend(cts)
        self.vars = None

    def register_constraint(self, ct):
        self.cts.append(ct)
        self.vars = None

    @staticmethod
    def _collect_constraint_variables(cts):
        # collect variables as a set
        var_set = set(v for c in cts for v in c.iter_variables())
        # convert to list
        var_list = list(var_set)
        var_list.sort(key=lambda dv: dv._index)
        return var_list

    def _get_or_collect_vars(self):
        if self.vars is None:
            self.vars = self._collect_constraint_variables(self.cts)
        return self.vars

    def make_solution(self):
        """ Creates and returns a DOcplex solution instance.

        This method should be called when CPLEX has a new incumbent solution.
        It stores variable values from the variables mentioned in the constraints.

        :return:
            An instance of SolveSolution.
        """
        return self.make_solution_from_vars(self._get_or_collect_vars())

    def get_cpx_unsatisfied_cts(self, cts, sol, tolerance=1e-6):
        """ returns the subset of unsatisfied constraints in a given solution.
        This is used in custom lazy constraints or user cut callbacks.

        :param cts: a list of constraints among which to look for unsatisfied
        :param sol: A solution object
        :param tolerance: amn optional numerical value used to determine
            whether a constraint is satisfied or not. Defaut is 1e-6.

        :return: a list of constraints from `cts`.
        """
        unsatisfied = []
        for ct in cts:
            if not ct.is_satisfied(sol, tolerance):
                # use mixin API to convert to cplex lingo
                cpx_lhs, cpx_sense, cpx_rhs = self.linear_ct_to_cplex(ct)
                # this add() method is specific to the type of CPLEX callback
                unsatisfied.append((ct, cpx_lhs, cpx_sense, cpx_rhs))
        return unsatisfied


def print_called(prompt_msg=None):
    """ A decorator function to be used on __call__() methods for derived callbacks.

    Use this decorator function to decorate __call__() methods of custom callbacks.

    Example:

    ::

        class MyCallback(ConstraintCallbackMixin, LazyConstraintCallback):

            @print_called('my custom callback called #{0}')
            def __call__(self):
                ...

        will print messages, before executing the callback code:

        >>> "my custom callback called #1"
        >>> "my custom callback called #2"

        each time the callback is called

    :param prompt_msg: A format string taking one argument (the number of calls)

    :return:
        As decorator, modifies the code of the __call_ method inplace.

    """

    def cb_decorator(func):
        prompt = prompt_msg or "* callback: {0} called: #{1}"

        def wrapper(self, *args, **kwargs):
            wrapper.count = wrapper.count + 1
            print(prompt.format(wrapper.count))
            res = func(self, *args, **kwargs)

            return res

        wrapper.count = 0
        return wrapper

    return cb_decorator

