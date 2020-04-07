# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2019
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module allows to solve a model expressed as a CPO file using a local solver
accessed through a shared library (.dll on Windows, .so on Linux).
Interface with library is done using ctypes module.
See https://docs.python.org/2/library/ctypes.html for details.
"""

from docplex.cp.solution import *
from docplex.cp.solution import CpoSolveResult
from docplex.cp.utils import CpoException, compare_natural
import docplex.cp.solver.solver as solver

import ctypes
from ctypes.util import find_library
import json
import sys
import time
import os


###############################################################################
## Constants
###############################################################################

# Events received from library
_EVENT_SOLVER_INFO   = 1  # Information on solver as JSON document
_EVENT_JSON_RESULT   = 2  # Solve result expressed as a JSON string
_EVENT_LOG_OUTPUT    = 3  # Log data on output stream
_EVENT_LOG_WARNING   = 4  # Log data on warning stream
_EVENT_LOG_ERROR     = 5  # Log data on error stream
_EVENT_SOLVER_ERROR  = 6  # Solver error. Details are in event associated string.


# Event notifier callback prototype
_EVENT_NOTIF_PROTOTYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)


###############################################################################
##  Public classes
###############################################################################

class CpoLibException(CpoException):
    """ The base class for exceptions raised by the library client
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(CpoLibException, self).__init__(msg)


###############################################################################
##  Public classes
###############################################################################

class CpoSolverLib(solver.CpoSolverAgent):
    """ Interface to a local solver through a shared library """
    __slots__ = ('libhdl',            # Lib handler
                 'session',           # Solve session in the library
                 'cbackproto',        # Prototype of the event callback
                 'first_error_line',  # First line of error
                 )

    def __init__(self, solver, params, context):
        """ Create a new solver using shared library.

        Args:
            solver:   Parent solver
            params:   Solving parameters
            context:  Solver agent context
        Raises:
            CpoLibException if library is not found
        """
        # Call super
        super(CpoSolverLib, self).__init__(solver, params, context)

        # Initialize attributes
        self.first_error_line = None
        self.libhdl = None # (to not block end() in case of init failure)

        # Connect to library
        self.libhdl = self._get_lib_handler()
        self.context.log(2, "Solving library: '{}'".format(self.context.lib))

        # Create session
        # CAUTION: storing callback prototype is mandatory. Otherwise, it is garbaged and the callback fails.
        self.cbackproto = _EVENT_NOTIF_PROTOTYPE(self._notify_event)
        self.session = self.libhdl.createSession(self.cbackproto)
        self.context.log(5, "Solve session: {}".format(self.session))

        # Check solver version if any
        sver = self.version_info.get('SolverVersion')
        mver = solver.get_model_format_version()
        if sver and mver and compare_natural(mver, sver) > 0:
            raise CpoLibException("Solver version {} is lower than model format version {}.".format(sver, mver))

        # Send CPO model to solver
        cpostr = self._get_cpo_model_string()
        self._set_cpo_model(cpostr)
        self.context.log(3, "Model set into solver.")


    def __del__(self):
        # End solve
        self.end()


    def solve(self):
        """ Solve the model

        According to the value of the context parameter 'verbose', the following information is logged
        if the log output is set:
         * 1: Total time spent to solve the model
         * 2: shared library file
         * 3: Main solving steps
         * 5: Every individual event got from library

        Returns:
            Model solve result (object of class CpoSolveResult)
        Raises:
            CpoException if error occurs
        """
        # Solve the model
        self._call_lib_function('solve', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def start_search(self):
        """ Start a new search. Solutions are retrieved using method search_next().
        """
        self._call_lib_function('startSearch', False)


    def search_next(self):
        """ Get the next available solution.

        Returns:
            Next model result (type CpoSolveResult)
        """
        # Call library function
        self._call_lib_function('searchNext', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def end_search(self):
        """ End current search.

        Returns:
            Last (fail) model solution with last solve information (type CpoSolveResult)
        """
        # Call library function
        self._call_lib_function('endSearch', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def abort_search(self):
        """ Abort current search.
        This method is designed to be called by a different thread than the one currently solving.
        """
        self._call_lib_function('abortSearch', False)


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        See documentation of CpoSolver.refine_conflict() for details.

        Returns:
            Conflict result,
            object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
        """
        # Ensure cpo model is generated with all constraints named
        if not self.context.model.name_all_constraints:
            self.context.model.name_all_constraints = True
            cpostr = self._get_cpo_model_string()
            self._set_cpo_model(cpostr)

        # Call library function
        self._call_lib_function('refineConflict', True)

        # Build result object
        return self._create_result_object(CpoRefineConflictResult, self.last_json_result)


    def propagate(self):
        """ This method invokes the propagation on the current model.

        See documentation of CpoSolver.propagate() for details.

        Returns:
            Propagation result,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        """
        # Call library function
        self._call_lib_function('propagate', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def run_seeds(self, nbrun):
        """ This method runs *nbrun* times the CP optimizer search with different random seeds
        and computes statistics from the result of these runs.

        This method does not return anything. Result statistics are displayed on the log output
        that should be activated.

        Each run of the solver is stopped according to single solve conditions (TimeLimit for example).
        Total run time is then expected to take *nbruns* times the duration of a single run.

        Args:
            nbrun: Number of runs with different seeds.
        Returns:
            Run result, object of class :class:`~docplex.cp.solution.CpoRunResult`.
        """
        # Call library function
        self._call_lib_function('runSeeds', False, nbrun)

        # Build result object
        self.last_json_result = None
        return self._create_result_object(CpoRunResult)


    def end(self):
        """ End solver and release all resources.
        """
        if self.libhdl is not None:
            self.libhdl.deleteSession(self.session)
            self.libhdl = None
            self.session = None
            super(CpoSolverLib, self).end()


    def _call_lib_function(self, dfname, json, *args):
        """ Call a library function
        Args:
            dfname:  Name of the function to be called
            json:    Indicate if a JSON result is expected
            *args:   Optional arguments (after session)
        Raises:
            CpoDllException if function call fails or if expected JSON is absent
        """
        # Reset JSON result if JSON required
        if json:
            self.last_json_result = None

        # Call library function
        rc = getattr(self.libhdl, dfname)(self.session, *args)
        if rc != 0:
            errmsg = "Call to '{}' failure (rc={})".format(dfname, rc)
            if self.first_error_line:
               errmsg += ": {}".format(self.first_error_line)
            raise CpoLibException(errmsg)

        # Check if JSON result is present
        if json and self.last_json_result is None:
            raise CpoLibException("No JSON result provided by function '{}'".format(dfname))


    def _notify_event(self, event, data):
        """ Callback called by the library to notify Python of an event (log, error, etc)
        Args:
            event:  Event id
            data:   Event data string
        """
        # Process event
        if event == _EVENT_LOG_OUTPUT or event == _EVENT_LOG_WARNING:
            # Store log if required
            if self.log_enabled:
                self._add_log_data(data.decode('utf-8'))

        elif event == _EVENT_JSON_RESULT:
            self.last_json_result = data.decode('utf-8')

        elif event == _EVENT_SOLVER_INFO:
            self.version_info = verinf = json.loads(data.decode('utf-8'))
            # Normalize information
            verinf['AgentModule'] = __name__
            iver = verinf.get('AngelVersion')
            if iver:
                verinf['InterfaceVersion'] = iver
            self.context.log(3, "Local solver info: '", verinf, "'")

        elif event == _EVENT_LOG_ERROR:
            ldata = data.decode('utf-8')
            if self.first_error_line is None:
                self.first_error_line = ldata.replace('\n', '')
            out = self.log_output if self.log_output is not None else sys.stdout
            out.write("ERROR: {}\n".format(ldata))
            out.flush()

        elif event == _EVENT_SOLVER_ERROR:
            errmsg = data.decode('utf-8')
            if self.first_error_line is not None:
                errmsg += " (" + self.first_error_line + ")"


    def _set_cpo_model(self, cpostr):
        """ Set the cpo model in the solver
        Args:
            cpostr:  CPO model as a string
        """
        # Encode model
        stime = time.time()
        cpostr = cpostr.encode('utf-8')
        self.process_infos.incr(CpoProcessInfos.MODEL_ENCODE_TIME, time.time() - stime)

        # Send CPO model to process
        stime = time.time()
        self._call_lib_function('setCpoModel', False, cpostr)
        self.process_infos.incr(CpoProcessInfos.MODEL_SEND_TIME, time.time() - stime)


    def _get_lib_handler(self):
        """ Access the CPO library
        Returns:
            Library handler, with function prototypes defined.
        Raises:
            CpoLibException if library is not found
        """
        # Access library
        listlibs = self.context.libfile
        if not listlibs:
            raise CpoLibException("CPO library file should be given in 'solver.lib.libfile' context attribute.")

        # Ensure it is a list of values
        if not is_array(listlibs):
            listlibs = (listlibs, )

        # Search the first available library
        lib = None
        for libf in listlibs[:-1]:
            try:
                lib = self._load_lib(libf)
                break
            except:
                pass
        # If not found, search last one, without catching exceptions
        if lib is None:
           lib = self._load_lib(listlibs[-1])

        # Define function prototypes
        for name, proto in _LIB_FUNCTION_PROTYTYPES.items():
            f = getattr(lib, name)
            f.restype = proto[0]
            f.argtypes = proto[1]

        # Return
        return lib


    def _load_lib(self, libf):
        """ Attempt to load a particular library
        Returns:
            Library handler
        Raises:
            CpoLibException or other Exception if library is not found
        """
        # Search for library file
        if not os.path.isfile(libf):
            lf = find_library(libf)
            if lf is None:
                raise CpoLibException("Can not find library '{}'".format(libf))
            libf = lf
        # Load library
        return ctypes.CDLL(libf)



###############################################################################
##  Private functions
###############################################################################

# Function prototypes
_LIB_FUNCTION_PROTYTYPES = \
{
    'createSession' : (ctypes.c_void_p, (ctypes.c_void_p,)),
    'deleteSession' : (ctypes.c_int,    (ctypes.c_void_p,)),
    'setCpoModel'   : (ctypes.c_int,    (ctypes.c_void_p, ctypes.c_char_p)),
    'solve'         : (ctypes.c_int,    (ctypes.c_void_p,)),
    'startSearch'   : (ctypes.c_int,    (ctypes.c_void_p,)),
    'searchNext'    : (ctypes.c_int,    (ctypes.c_void_p,)),
    'endSearch'     : (ctypes.c_int,    (ctypes.c_void_p,)),
    'abortSearch'   : (ctypes.c_int,    (ctypes.c_void_p,)),
    'propagate'     : (ctypes.c_int,    (ctypes.c_void_p,)),
    'refineConflict': (ctypes.c_int,    (ctypes.c_void_p,)),
    'runSeeds'      : (ctypes.c_int,    (ctypes.c_void_p, ctypes.c_int,)),
}



