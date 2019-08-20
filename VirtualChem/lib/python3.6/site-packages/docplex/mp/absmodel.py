# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.format import LP_format
from docplex.mp.model import Model


class _AbstractModelMixin(object):

    def setup_variables(self):
        raise NotImplementedError  # pragma: no cover

    def setup_constraints(self):
        raise NotImplementedError  # pragma: no cover

    def setup_objective(self):
        ''' Redefine this method to set the objective.
        This is not mandatory as a model might not have any objective.
        '''
        pass  # pragma: no cover

    # noinspection PyMethodMayBeStatic
    def check(self):
        ''' Redefine this method to check the model before solve.
        '''
        pass

    def setup_data(self):
        pass

    def post_process(self):
        pass

    def setup(self):
        """ Setup the model artifacts, raise exception if data are not correct.
        """
        self.setup_data()
        self.setup_variables()
        self.setup_constraints()
        self.setup_objective()

    def ensure_setup(self):
        if self._is_empty():
            self.setup()

    def before_solve_hook(self):
        """ This method is called just before solve inside a run.
        Redefine to get some particular behavior.
        """
        pass

    def export(self, path=None, basename=None, hide_user_names=False, exchange_format=LP_format):
        # INTERNAL: redefine export at this stage to ensure model is setup
        self.ensure_setup()
        return super(_AbstractModelMixin, self).export(path, basename, hide_user_names, exchange_format)

    def run_silent(self, **kwargs):
        # make sure the model is setup
        self.ensure_setup()
        # check data and model if necessary (default is do nothing)
        self.check()
        # insert some last minute code before solve.
        self.before_solve_hook()
        # call solve_run which by default calls solve
        s = self.solve_run(**kwargs)
        if s:
            self.post_process()
        return s

    def solve_run(self, **kwargs):
        return self.solve(**kwargs)

    def run(self, **kwargs):
        s = self.run_silent(**kwargs)
        if s:
            self.report()
        return s


class AbstractModel(_AbstractModelMixin, Model):
    def __init__(self, name, context=None, **kwargs):
        Model.__init__(self, name=name, context=context, **kwargs)
