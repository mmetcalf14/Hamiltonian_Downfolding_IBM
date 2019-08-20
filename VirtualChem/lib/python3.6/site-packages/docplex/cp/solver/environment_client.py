# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016, 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module contains the client that allows to notify solving environment
with relevant events.

The solving environment is typically local or a Python worker.

Real implementation of environment specifics is done in module
docplex.util.environment.py. The present module provides what is necessary
to call it with appropriate CPO solver data.

Note tha ta default null behavior is provided if the docplex.util.environment
can not be imported.
"""

from docplex.cp.solution import *
from docplex.cp.solver.solver_listener import CpoSolverListener
import json

try:
    import docplex.util.environment as runenv
except:
    runenv = None


#==============================================================================
# Constants
#==============================================================================

# Possible solve statuses
_STATUS_UNKNOWN                          = 0  # The algorithm has no information about the solution.
_STATUS_FEASIBLE_SOLUTION                = 1  # The algorithm found a feasible solution.
_STATUS_OPTIMAL_SOLUTION                 = 2  # The algorithm found an optimal solution.
_STATUS_INFEASIBLE_SOLUTION              = 3  # The algorithm proved that the model is infeasible.
_STATUS_UNBOUNDED_SOLUTION               = 4  # The algorithm proved the model unbounded.
_STATUS_INFEASIBLE_OR_UNBOUNDED_SOLUTION = 5  # The model is infeasible or unbounded.

# Map of CPO solve status on environment status
_SOLVE_STATUS_MAP = {SOLVE_STATUS_FEASIBLE   : _STATUS_FEASIBLE_SOLUTION,
                     SOLVE_STATUS_INFEASIBLE : _STATUS_INFEASIBLE_SOLUTION,
                     SOLVE_STATUS_OPTIMAL    : _STATUS_OPTIMAL_SOLUTION}


#==============================================================================
# Classes
#==============================================================================

# Solver listener that interact with environment
class EnvSolverListener(CpoSolverListener):
    """ Cpo solver listener that interact with environment.
    This listener is added by the CpoSolver when it is created, if the environment exists.
    """
    __slots__ = ('env',               # Environment
                 'publish_context',   # Solver publish context
                 )

    def solver_created(self, solver):
        """ Notify the listener that the solver object has been created.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        # Init listener
        if not self._init_listener(solver):
            # Remove this listener from solver
            solver.remove_listener(self)
            return

        # Check if calling environment is DODS (Decision Optimization for Data Science)
        if self.env.is_dods():
            # Force solve() method to proceed with start()/next()
            solver.context.solver.solve_with_start_next = True
        # Check if debug mode is required
        if self.env.is_debug_mode():
            # Force more debug information
            solver.context.params.WarningLevel = 3
            solver.context.params.PrintModelDetailsInMessages = "On"


    def start_solve(self, solver):
        """ Notify that the solve is started.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        if self.publish_context.get_attribute('solve_details', True):
            # Set ordered list of KPIs in solve details
            kpis = solver.get_model().get_kpis()
            if kpis:
                # Add ordered list of kpi names
                sdetails = {'MODEL_DETAIL_KPIS': json.dumps(list(kpis.keys()))}
                self.env.notify_start_solve(sdetails)


    def end_solve(self, solver):
        """ Notify that the solve is ended.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        if self.publish_context.get_attribute('solve_details', True):
            res = solver.get_last_result()
            if res is None:
                status = _STATUS_UNKNOWN
            else:
                status = _SOLVE_STATUS_MAP.get(res.get_solve_status(), _STATUS_UNKNOWN)
            self.env.notify_end_solve(status)


    def result_found(self, solver, msol):
        """ Signal that a solution has been found.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
            msol:   Model solution, object of class :class:`~docplex.cp.solution.CpoSolveResult`
        """
        # Get last solution
        if msol is None or not msol:
            return

        # Publish solve details
        if self.publish_context.get_attribute('solve_details', True):
            self.publish_context.log(2, "Publish solve details")

            # Build solve details
            infos = msol.get_infos()
            sdetails = {}
            nbintvars = infos.get("NumberOfIntegerVariables")
            if nbintvars is not None:
                sdetails["MODEL_DETAIL_INTEGER_VARS"] = nbintvars
            nbintervars = infos.get("NumberOfIntervalVariables")
            if nbintervars is not None:
                sdetails["MODEL_DETAIL_INTERVAL_VARS"] = nbintervars
            nbseqvars = infos.get("NumberOfSequenceVariables")
            if nbseqvars is not None:
                sdetails["MODEL_DETAIL_SEQUENCE_VARS"] = nbseqvars
            nbconstr = infos.get("NumberOfConstraints")
            if nbconstr is not None:
                sdetails["MODEL_DETAIL_CONSTRAINTS"] = nbconstr
            # Set detail type
            if (nbintervars in (0, None)) and (nbseqvars in (0, None)):
                sdetails["MODEL_DETAIL_TYPE"] = "CPO CP"
            else:
                sdetails["MODEL_DETAIL_TYPE"] = "CPO Scheduling"

            # Set objective if any
            objctv = msol.get_objective_values()
            if objctv is not None:
                sdetails["PROGRESS_CURRENT_OBJECTIVE"] = ';'.join([str(x) for x in objctv])

            # Set KPIs if any
            kpis = msol.get_kpis()
            if kpis:
                # Add ordered list of kpi names
                sdetails["MODEL_DETAIL_KPIS"] = json.dumps(list(kpis.keys()))
                # Add KPIs
                for k, v in kpis.items():
                    sdetails["KPI." + k] = v

            # Submit details to environment
            self.publish_context.log(3, "Solve details: ", sdetails)
            self.env.update_solve_details(sdetails)

        # Write JSON solution as output
        resout = self.publish_context.get_attribute('result_output', 'solution.json')
        if resout:
            jdoc = solver.agent._get_last_json_result_string()
            if jdoc is not None:
                self.publish_context.log(2, "Publish JSON result output in '", resout, "'")
                with self.env.get_output_stream(resout) as fp:
                    fp.write(jdoc.encode('utf-8'))

        # Publish kpis
        kpiout = self.publish_context.get_attribute('kpis_output', 'kpis.csv')
        if kpiout:
            kpis = msol.get_kpis()
            if kpis:
                self.publish_context.log(2, "Publish KPIs result output in '", kpiout, "'")
                with self.env.get_output_stream(kpiout) as fp:
                    fp.write('"NAME","VALUE"\n'.encode('utf-8'))
                    for k, v in kpis.items():
                        fp.write('{},{}\n'.format(encode_csv_string(k), v).encode('utf-8'))


    def conflict_found(self, solver, cflct):
        """ Signal that a conflict has been found.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
            cflct:  Conflict descriptor, object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`
        """
        # Publish conflict
        cfltout = self.publish_context.get_attribute('conflicts_output', 'conflicts.csv')
        if cfltout:
            self.publish_context.log(2, "Publish conflict output in '", cfltout, "'")
            with self.env.get_output_stream(cfltout) as fp:
                fp.write('"Type","Status","Name","Expression"\n'.encode('utf-8'))
                for c in cflct.get_all_member_constraints():
                    fp.write(self._build_conflict_csv_constraint_line(c, "Member").encode('utf-8'))
                for c in cflct.get_all_possible_constraints():
                    fp.write(self._build_conflict_csv_constraint_line(c, "Possible_member").encode('utf-8'))
                for v in cflct.get_all_member_variables():
                    fp.write(self._build_conflict_csv_variable_line(v, "Member").encode('utf-8'))
                for v in cflct.get_all_possible_variables():
                    fp.write(self._build_conflict_csv_variable_line(v, "Possible_member").encode('utf-8'))


    def _init_listener(self, solver):
        """ Initialize this listener
        Args:
            solver:  Calling solver
        Returns:
            True if listener is OK, False if should be removed from solver.
        """
        # Check if environment package not present
        env = get_environment()
        if env is None:
            return False
        self.env = env

        # Retrieve auto_publish context
        pctx = solver.context.solver.auto_publish
        if pctx is True:
            # Create default context to retrieve always default values
            pctx = Context()
        elif not isinstance(pctx, Context):
            return False
        self.publish_context = pctx

        # Check no local publish
        if isinstance(env, runenv.LocalEnvironment) and not pctx.local_publish:
            return False

        # Keep as listener
        return True



    def _build_conflict_csv_constraint_line(self, cstr, csts):
        """ Build the string representing a constraint in conflicts csv
        Args:
            cstr:  Constraint
            csts:  Contraint status
        Returns:
            String representing a constraint line in conflicts csv
        """
        # Build constraint elements
        name = cstr.get_name()
        res = '{},{},{},{}\n'.format(encode_csv_string('Constraint'),
                                     encode_csv_string(csts),
                                     encode_csv_string(str(name)) if name is not None else '""',
                                     encode_csv_string(str(cstr)[:100]))
        return res


    def _build_conflict_csv_variable_line(self, var, vsts):
        """ Build the string representing a constraint in conflicts csv
        Args:
            var:   Variable
            vsts:  Variable status
        Returns:
            String representing a variable line in conflicts csv
        """
        # Build constraint elements
        name = var.get_name()
        res = '{},{},{},{}\n'.format(encode_csv_string('Variable'),
                                     encode_csv_string(vsts),
                                     encode_csv_string(str(name)) if name is not None else '""',
                                     encode_csv_string(var))
        return res


#==============================================================================
# Public functions
#==============================================================================


def get_environment():
    """ Get the environment descriptor

    Returns:
        Environment descriptor, None if none
    """
    return None if runenv is None else runenv.get_environment()


def is_environment_present():
    """ Check whether environment is present
    Returns:
        True if environment is present, false otherwise
    """
    return get_environment() is not None


def new_solver_created(solver):
    """ Add environment solver listener if needed
    Args:
        solver:  Solver to update
    """
    # Check if environment package not present
    if runenv is None:
        return

    # Check no environment
    env = runenv.get_environment()
    if env is None:
        return

    # Retrieve auto_publish context
    pctx = solver.context.solver.auto_publish
    if pctx is True:
        # Create default context to retrieve always default values
        pctx = Context()
    elif not isinstance(pctx, Context):
        return

    # Check no local publish
    if isinstance(env, runenv.LocalEnvironment) and not pctx.local_publish:
        return

    # Add solver listener
    solver.add_listener(EnvSolverListener(env, pctx))


