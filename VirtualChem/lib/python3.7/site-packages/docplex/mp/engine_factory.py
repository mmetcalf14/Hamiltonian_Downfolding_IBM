# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.engine import NoSolveEngine, ZeroSolveEngine, FakeFailEngine, RaiseErrorEngine
from docplex.mp.docloud_engine import DOcloudEngine
from docplex.mp.utils import DOcplexException, is_string
from docplex.mp.context import has_credentials

class EngineFactory(object):
    """ A factory class that manages creation of solver instances.
    """
    _default_engine_map = {"local": NoSolveEngine,
                           "nosolve": NoSolveEngine,
                           "zero": ZeroSolveEngine,
                           "fail": FakeFailEngine,
                           "raise": RaiseErrorEngine,
                           "docloud": DOcloudEngine}

    cplex_engine_type = None

    def __init__(self, env=None):
        self._engine_types_by_agent = self._default_engine_map.copy()
        # no cplex engine type yet?
        if env is not None:
            self._resolve_cplex(env)

    def _get_engine_from_agent(self, agent, default_engine, default_engine_name):
        if agent is None:
            return default_engine
        elif is_string(agent):
            agent_key = agent.lower()
            engine_type = self._engine_types_by_agent.get(agent_key)
            if engine_type:
                return engine_type
            elif 'cplex' == agent_key:
                print('* warning: CPLEX DLL not found in path, using {0} instead'.format(default_engine_name))

            else:
                print('* warning: unrecognized solver agent value: {0!r}'.format(agent))

        else:
            print('* warning: incorrect solver agent value: {0!r} -expecting string or None'.format(agent))

        return default_engine

    def _is_cplex_resolved(self):
        return hasattr(self, "_cplex_engine_type")

    def _resolve_cplex(self, env):
        # INTERNAL
        if env is None:
            raise DOcplexException("need an environment to resolve cplex, got None")
        if not self._is_cplex_resolved():
            if env.has_cplex:
                from docplex.mp.cplex_engine import CplexEngine

                self._cplex_engine_type = CplexEngine
                self._engine_types_by_agent["cplex"] = CplexEngine
            else:
                self._cplex_engine_type = None

    def _ensure_cplex_resolved(self, env):
        if not self._is_cplex_resolved():
            self._resolve_cplex(env)
        assert self._is_cplex_resolved()


    def new_engine(self, solver_agent, env, model, context=None):
        self._ensure_cplex_resolved(env)

        # compute a default engine and kwargs to use..
        kwargs = {}
        if self._cplex_engine_type:
            # CPLEX ahs been resolved and has a non-None type
            # default is CPLEX if we have it
            default_engine_type = self._cplex_engine_type
            default_engine_name = 'cplex'

        elif has_credentials(context.solver.docloud):
            # default is docloud
            default_engine_type = DOcloudEngine
            default_engine_name = 'docloud'

        else:
            # no CPLEX, no credentials
            # model.trace("CPLEX DLL not found and model has no DOcplexcloud credentials. "
            # "Credentials are required at solve time")
            default_engine_type = NoSolveEngine
            default_engine_name = 'nosolve'

        if has_credentials(context.solver.docloud):
            kwargs['docloud_context'] = context.solver.docloud

        engine_type = self._get_engine_from_agent(agent=solver_agent,
                                                  default_engine=default_engine_type,
                                                  default_engine_name=default_engine_name)
        assert engine_type is not None
        # all engine types have a (model, kwargs) ctor.
        return engine_type(model, **kwargs)

    # noinspection PyMethodMayBeStatic
    def new_docloud_engine(self, model, **kwargs):
        return DOcloudEngine(model, **kwargs)

    def extend(self, new_agent, new_engine):
        # INTERNAL
        assert new_engine is not None
        self._engine_types_by_agent[new_agent] = new_engine


