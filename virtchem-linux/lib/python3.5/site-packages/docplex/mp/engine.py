# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


from docplex.mp.solution import SolveSolution
from docplex.mp.sdetails import SolveDetails
from docplex.mp.utils import DOcplexException
from docplex.util.status import JobSolveStatus
from docplex.mp.constants import CplexScope


# gendoc: ignore


class ISolver(object):
    """
    The pure solving part
    """

    def can_solve(self):
        """
        :return: True if this engine class can truly solve
        """
        raise NotImplementedError  # pragma: no cover

    def register_callback(self, cb):
        raise NotImplementedError  # pragma: no cover

    def connect_progress_listeners(self, listeners, model):
        """
        Connects progress listeners
        :param listeners:
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def solve(self, mdl, parameters, lex_mipstart=None, lex_timelimits=None, lex_mipgaps=None):
        ''' Redefine this method for the real solve.
            Returns a solution object or None.
        '''
        raise NotImplementedError  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        """
        Runs feasopt-like algorithm with a set of relaxable cts with preferences
        :param relaxable_groups:
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        """
        Runs conflict-refiner algorithm with an optional set constraints groups with preferences

        :param mdl: The model for which conflict refinement is performed.
        :param preferences: an optional dictionary defining constraints preferences.
        :param groups: an optional list of 'docplex.mp.conflict_refiner.ConstraintsGroup'.
        :param parameters:
        :return: A list of "TConflictConstraint" namedtuples, each tuple corresponding to a constraint that is
            involved in the conflict.
            The fields of the "TConflictConstraint" namedtuple are:
                - the name of the constraint or None if the constraint corresponds to a variable lower or upper bound
                - a reference to the constraint or to a wrapper representing a Var upper or lower bound
                - an :enum:'docplex.mp.constants.ConflictStatus' object that indicates the
                conflict status type (Excluded, Possible_member, Member...)
            This list is empty if no conflict is found by the conflict refiner.
        """
        raise NotImplementedError  # pragma: no cover

    def get_solve_status(self):
        """  Return a DOcplexcloud-style solve status.

        Possible enums are in docloud/status.py
        Default is UNKNOWN at this stage. Redefined for CPLEX and DOcplexcloud engines.
        """
        return JobSolveStatus.UNKNOWN  # pragma: no cover

    def get_cplex(self):
        """
        Returns the underlying CPLEX, if any. May raise an exception if not applicable.
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def has_cplex(self):  # pragma: no cover
        try:
            return self.get_cplex() is not None
        except DOcplexException:
            # some engine may raise an exception when accessing a cplex
            return False

    def set_parameter(self, parameter, value):
        """ Changes the parameter value.
        :param parameter:
        :param value:
        """
        raise NotImplementedError  # pragma: no cover

    def get_parameter(self, parameter):
        raise NotImplementedError  # pragma: no cover

    def get_solve_details(self):
        raise NotImplementedError  # pragma: no cover

    def get_quality_metrics(self):
        raise NotImplementedError  # pragma: no cover

    def clean_before_solve(self):
        raise NotImplementedError  # pragma: no cover

    def supports_logical_constraints(self):
        raise NotImplementedError  # pragma: no cover

    def solved_as_mip(self):
        return False


# noinspection PyAbstractClass
class IEngine(ISolver):
    """ interface for all engine facades
    """

    def get_name(self):
        ''' Returns the code to be used in model'''
        raise NotImplementedError  # pragma: no cover

    @property
    def name(self):
        return self.get_name()

    def get_var_index(self, var):
        raise NotImplementedError  # pragma: no cover

    def get_ct_index(self, index):
        raise NotImplementedError  # pragma: no cover

    def get_infinity(self):
        raise NotImplementedError  # pragma: no cover

    def create_one_variable(self, vartype, lb, ub, name):
        raise NotImplementedError  # pragma: no cover

    def create_variables(self, keys, vartype, lb, ub, name):
        raise NotImplementedError  # pragma: no cover

    def create_multitype_variables(self, keys, vartypes, lbs, ubs, names):
        raise NotImplementedError  # pragma: no cover

    def create_linear_constraint(self, binaryct):
        raise NotImplementedError  # pragma: no cover

    def create_block_linear_constraints(self, ct_seq):
        raise NotImplementedError  # pragma: no cover

    def create_range_constraint(self, rangect):
        raise NotImplementedError  # pragma: no cover

    def create_indicator_constraint(self, ind):
        raise NotImplementedError  # pragma: no cover

    def create_equivalence_constraint(self, eqct):
        raise NotImplementedError  # pragma: no cover

    def create_batch_equivalence_constraints(self, eqcts):
        # the default is to iterate and append.
        return [self.create_equivalence_constraint(eqc) for eqc in eqcts]

    def create_batch_indicator_constraints(self, inds):
        return [self.create_indicator_constraint(ind) for ind in inds]

    def create_quadratic_constraint(self, qct):
        raise NotImplementedError  # pragma: no cover

    def create_pwl_constraint(self, pwl_ct):
        raise NotImplementedError  # pragma: no cover

    def remove_constraint(self, ct):
        raise NotImplementedError  # pragma: no cover

    def remove_constraints(self, cts):
        raise NotImplementedError  # pragma: no cover

    def set_objective_sense(self, sense):
        raise NotImplementedError  # pragma: no cover

    def set_objective_expr(self, new_objexpr, old_objexpr):
        raise NotImplementedError  # pragma: no cover

    def set_multi_objective_exprs(self, new_multiobjexprs, old_multiobjexprs,
                                  priorities=None, weights=None, abstols=None, reltols=None, objnames=None):
        raise NotImplementedError  # pragma: no cover

    def end(self):
        raise NotImplementedError  # pragma: no cover

    def set_streams(self, out):
        raise NotImplementedError  # pragma: no cover

    def set_var_lb(self, var, lb):
        raise NotImplementedError  # pragma: no cover

    def set_var_ub(self, var, ub):
        raise NotImplementedError  # pragma: no cover

    def rename_var(self, var, new_name):
        raise NotImplementedError  # pragam: no cover

    def set_var_type(self, var, new_type):
        raise NotImplementedError  # pragam: no cover

    def update_objective_epxr(self, expr, event, *args):
        raise NotImplemented  # pragma: no cover

    def update_constraint(self, ct, event, *args):
        raise NotImplementedError  # pragma: no cover

    def check_var_indices(self, dvars):
        raise NotImplementedError  # pragma: no cover

    def check_constraint_indices(self, cts):
        raise NotImplementedError  # pragma: no cover

    def create_sos(self, sos):
        raise NotImplementedError  # pragma: no cover

    def clear_all_sos(self):
        raise NotImplementedError  # pragma: no cover

    def get_basis(self, mdl):
        raise NotImplementedError  # pragma: no cover

    def set_lp_start(self, var_stats, ct_stats):
        raise NotImplementedError

# noinspection PyAbstractClass
class DummyEngine(IEngine):
    def create_range_constraint(self, rangect):
        return -1  # pragma: no cover

    def create_indicator_constraint(self, ind):
        return -1  # pragma: no cover

    def create_equivalence_constraint(self, eqct):
        return -1  # pragma: no cover

    def create_quadratic_constraint(self, qct):
        return -1  # pragma: no cover

    def create_pwl_constraint(self, pwl_ct):
        return -1  # pragma: no cover

    def set_streams(self, out):
        pass  # pragma: no cover

    def get_infinity(self):
        return 1e+20  # pragma: no cover

    def get_var_index(self, var):
        return -1  # pragma: no cover

    def get_ct_index(self, index):
        return -1  # pragma: no cover

    def create_one_variable(self, vartype, lb, ub, name):
        return -1  # pragma: no cover

    def create_variables(self, keys, vartype, lb, ub, name):
        return [-1] * len(keys)  # pragma: no cover

    def create_multitype_variables(self, keys, vartypes, lbs, ubs, names):
        return [-1] * len(keys)

    def set_var_lb(self, var, lb):
        pass

    def set_var_ub(self, var, ub):
        pass

    def rename_var(self, var, new_name):
        pass  # nothing to do, except in cplex...

    def rename_linear_constraint(selfself, linct, new_name):
        pass # nothing to do, except in cplex...

    def set_var_type(self, var, new_type):
        pass  # nothing to do, except in cplex...

    def create_linear_constraint(self, binaryct):
        return -1  # pragma: no cover

    def create_block_linear_constraints(self, ct_seq):
        return [-1] * len(ct_seq)  # pragma: no cover

    def create_batch_indicator_constraints(self, ind_seq):
        return [-1] * len(ind_seq)  # pragma: no cover

    def remove_constraint(self, ct):
        pass  # pragma: no cover

    def remove_constraints(self, cts):
        pass  # pragma: no cover

    def set_objective_sense(self, sense):
        pass  # pragma: no cover

    def set_objective_expr(self, new_objexpr, old_objexpr):
        pass  # pragma: no cover

    def set_multi_objective_exprs(self, new_multiobjexprs, old_multiobjexprs,
                                  priorities=None, weights=None, abstols=None, reltols=None, objnames=None):
        pass  # pragma: no cover

    def end(self):
        """ terminate the engine
        """
        pass  # pragma: no cover

    def register_callback(self, cb):
        pass  # pragma: no cover

    def unregister_callback(self, cb):
        pass  # pragma: no cover

    def connect_progress_listeners(self, listeners, model):
        if listeners:
            model.warning("Progress listeners require CPLEX, not supported on engine {0}.".format(self.name))

    def disconnect_progress_listeners(self, listeners):
        pass  # pragma: no cover

    def can_solve(self):
        return False  # pragma: no cover

    def solve(self, mdl, parameters, lex_mipstart=None, lex_timelimits=None, lex_mipgaps=None):
        return None  # pragma: no cover

    def get_solve_status(self):
        return JobSolveStatus.UNKNOWN  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        raise None  # pragma: no cover

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        raise None  # pragma: no cover

    def get_cplex(self):
        raise DOcplexException("No CPLEX is available.")  # pragma: no cover

    def clean_before_solve(self):
        pass  # pragma: no cover

    def update_objective(self, expr, event, *args):
        # nothing to do except for cplex
        pass  # pragma: no cover


    def update_constraint(self, ct, event, *args):
        pass  # pragma: no cover

    def get_quality_metrics(self):
        return {}  # pragma: no cover

    def supports_logical_constraints(self):
        return True, None

    def supports_multi_objective(self):
        return True, None

    def check_var_indices(self, dvars):
        pass

    def check_constraint_indices(self, cts):
        pass

    def create_sos(self, sos):
        pass

    def clear_all_sos(self):
        pass


    def get_basis(self, mdl):
        return None, None

    def set_lp_start(self, var_stats, ct_stats):
        raise DOcplexException('set_lp_start() requires CPLEX, not available for {0}'.format(self.name))


# noinspection PyAbstractClass,PyUnusedLocal
class IndexerEngine(DummyEngine):
    """
    An abstract engine facade which generates unique indices for variables, constraints
    """

    def __init__(self, initial_index=0):
        DummyEngine.__init__(self)
        self._initial_index = initial_index  # CPLEX indices start at 0, not 1
        self.__var_counter = self._initial_index
        self._ct_counter = self._initial_index

    def _increment_vars(self, size=1):
        self.__var_counter += size
        return self.__var_counter

    def _increment_cts(self, size=1):
        self._ct_counter += size
        return self._ct_counter

    def create_one_variable(self, vartype, lb, ub, name):
        old_count = self.__var_counter
        self._increment_vars(1)
        return old_count

    def create_variables(self, keys, vartype, lb, ub, name):
        old_count = self.__var_counter
        new_count = self._increment_vars(len(keys))
        return list(range(old_count, new_count))

    def create_multitype_variables(self, keys, vartypes, lbs, ubs, names):
        old_count = self.__var_counter
        new_count = self._increment_vars(len(keys))
        return list(range(old_count, new_count))

    def _create_one_ct(self):
        old_ct_count = self._ct_counter
        self._increment_cts(1)
        return old_ct_count

    def create_linear_constraint(self, binaryct):
        return self._create_one_ct()

    def create_batch_cts(self, ct_seq):
        old_ct_count = self._ct_counter
        size = sum(1 for _ in ct_seq) # iterator is consumed
        self._increment_cts(size)
        return range(old_ct_count, self._ct_counter)

    def create_block_linear_constraints(self, ct_seq):
        return self.create_batch_cts(ct_seq)

    def create_range_constraint(self, rangect):
        return self._create_one_ct()

    def create_indicator_constraint(self, ind):
        return self._create_one_ct()

    def create_equivalence_constraint(self, eqct):
        return self._create_one_ct()

    def create_batch_indicator_constraints(self, indicators):
        return self.create_batch_cts(indicators)

    def create_quadratic_constraint(self, ind):
        return self._create_one_ct()

    def create_pwl_constraint(self, pwl_ct):
        return self._create_one_ct()

    def get_all_reduced_costs(self, mdl):
        return {}

    def get_all_dual_values(self, mdl):
        return {}

    def get_all_slack_values(self, mdl):
        return {CplexScope.LINEAR_CT_SCOPE: {},
                CplexScope.QUAD_CT_SCOPE: {},
                CplexScope.IND_CT_SCOPE: {}}

    def dump(self, path):
        pass

    def set_objective_sense(self, sense):
        pass

    def set_objective_expr(self, new_objexpr, old_objexpr):
        pass

    def set_multi_objective_exprs(self, new_multiobjexprs, old_multiobjexprs,
                                  priorities=None, weights=None, abstols=None, reltols=None, objnames=None):
        pass

    def set_parameter(self, parameter, value):
        """ Changes the parameter value in the engine.

        For this limited type of engine, nothing to do.

        """
        pass

    def get_parameter(self, parameter):
        """ Gets the current value of a parameter.

        Params:
         parameter: the parameter for which we query the value.

        """
        return parameter.get()


class NoSolveEngine(IndexerEngine):
    def get_solve_details(self):
        SolveDetails.make_fake_details(time=0, feasible=False)

    # INTERNAL: a dummy engine that cannot solve.

    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)

    def get_name(self):
        return "local"

    def get_var_index(self, var):
        return var.index

    def get_ct_index(self, ct):
        return ct.index

    def can_solve(self):
        return False

    def solve(self, mdl, parameters, lex_mipstart=None, lex_timelimits=None, lex_mipgaps=None):
        """
        This solver cannot solve. never ever.
        """
        mdl.fatal("No CPLEX DLL and no DOcplexcloud credentials: model cannot be solved!")
        return None

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        mdl.fatal("No CPLEX DLL: model cannot be relaxed!")
        return None

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        mdl.fatal("No CPLEX DLL: conflict refiner cannot be invoked on model!")
        return None

    @staticmethod
    def make_from_model(mdl):
        eng = NoSolveEngine(mdl)
        eng._increment_vars(mdl.number_of_variables)
        eng._increment_cts(mdl.number_of_constraints)
        return eng


class ZeroSolveEngine(IndexerEngine):
    # INTERNAL: a dummy engine that says it can solve
    # but returns an all-zero solution.
    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)  # pragma: no cover
        self._last_solved_parameters = None

    def show_parameters(self, params):
        if params is None:
            print("DEBUG> parameters: None")
        else:
            if params.has_nondefaults():
                print("DEBUG> parameters:")
                params.print_information(indent_level=8)  #
            else:
                print("DEBUG> parameters: defaults")

    @property
    def last_solved_parameters(self):
        return self._last_solved_parameters

    def get_name(self):
        return "zero_solve"  # pragma: no cover

    @property
    def name(self):
        return self.get_name()

    def get_var_zero_solution(self, dvar):
        return max(0, dvar.lb)

    def solve(self, mdl, parameters, lex_mipstart=None, lex_timelimits=None, lex_mipgaps=None):
        # remember last solved params
        self._last_solved_parameters = parameters.clone() if parameters is not None else None
        self.show_parameters(parameters)
        return self.make_zero_solution(mdl)

    def get_solutions(self, args):
        if not args:
            return {}
        else:
            return {v: 0 for v in args}

    def can_solve(self):
        return True  # pragma: no cover

    def make_zero_solution(self, mdl):
        # return a feasible value: max of zero and the lower bound
        zlb_map = {v: self.get_var_zero_solution(v) for v in mdl.iter_variables() if v.lb != 0}
        obj = mdl.objective_expr.constant
        return SolveSolution(mdl, obj=obj, var_value_map=zlb_map, solved_by=self.name)  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        params = parameters or mdl.parameters
        self._last_solved_parameters = params
        self.show_parameters(params)
        return self.make_zero_solution(mdl)

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        return None

    def get_solve_details(self):
        return SolveDetails.make_fake_details(time=0, feasible=True)


class FakeFailEngine(IndexerEngine):
    # INTERNAL: a dummy engine that says it can solve
    # but always fail, and returns None.
    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)  # pragma: no cover

    def get_name(self):
        return "no_solution_solve"  # pragma: no cover

    def solve(self, mdl, parameters, lex_mipstart=None, lex_timelimits=None, lex_mipgaps=None):
        # solve fails equivalent to returning None
        return None  # pragma: no cover

    def can_solve(self):
        return True  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        return None  # pragma: no cover

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        return None  # pragma: no cover

    def get_solve_status(self):
        return JobSolveStatus.INFEASIBLE_SOLUTION  # pragma: no cover

    def get_solve_details(self):
        return SolveDetails.make_fake_details(time=0, feasible=False)


class TerminatedEngine(IndexerEngine):
    # INTERNAL: a dummy engine that says it can solve
    # but always fail, and returns None.
    def terminate(self):
        raise DOcplexException("model has been terminated, no solve is possible...")

    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)  # pragma: no cover

    def name(self):
        return "exception"  # pragma: no cover

    def solve(self, mdl, parameters, lex_mipstart=None, lex_timelimits=None, lex_mipgaps=None):
        # solve fails equivalent to returning None
        self.terminate()
        return None  # pragma: no cover

    def can_solve(self):
        return True  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        self.terminate()
        return None  # pragma: no cover

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        self.terminate()
        return None  # pragma: no cover

    def get_solve_status(self):
        return JobSolveStatus.INFEASIBLE_SOLUTION  # pragma: no cover

    def get_solve_details(self):
        return SolveDetails.make_fake_details(time=0, feasible=False)


class RaiseErrorEngine(IndexerEngine):
    # INTERNAL: a dummy engine that says it can solve
    # but always raises an exception, this is for testing

    @staticmethod
    def _simulate_error():
        raise DOcplexException("simulate exception")

    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)  # pragma: no cover

    def get_name(self):
        return "raise"  # pragma: no cover

    def solve(self, mdl, parameters, lex_mipstart=None, lex_timelimits=None, lex_mipgaps=None):
        # solve fails equivalent to returning None
        self._simulate_error()
        return None  # pragma: no cover

    def can_solve(self):
        return True  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        self._simulate_error()
        return None  # pragma: no cover

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        self._simulate_error()
        return None  # pragma: no cover

    def get_solve_status(self):
        return JobSolveStatus.INFEASIBLE_SOLUTION  # pragma: no cover

    def get_solve_details(self):
        return SolveDetails.make_fake_details(time=0, feasible=False)
