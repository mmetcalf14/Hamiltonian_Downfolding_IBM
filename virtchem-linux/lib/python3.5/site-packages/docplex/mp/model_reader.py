# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

import os

# docplex
from docplex.mp.model import Model
from docplex.mp.utils import DOcplexException

from docplex.mp.params.cplex_params import get_params_from_cplex_version
from docplex.mp.constants import ComparisonType

# cplex
try:
    from cplex import Cplex
    from cplex._internal._subinterfaces import ObjSense
    from cplex.exceptions import CplexError, CplexSolverError
    from docplex.mp.cplex_engine import _safe_cplex

except ImportError:  # pragma: no cover
    Cplex = None

from docplex.mp.compat23 import izip
from docplex.mp.quad import VarPair


class ModelReaderError(DOcplexException):
    pass


class _CplexReaderFileContext(object):
    def __init__(self, filename, read_method=None):
        self._cplex = None
        self._filename = filename
        self._read_method = read_method or ["read"]

    def __enter__(self):
        cpx = _safe_cplex()
        # no output from CPLEX
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.set_warning_stream(None)
        cpx.set_error_stream(None)
        self_read_fn = cpx
        for m in self._read_method:
            self_read_fn = self_read_fn.__getattribute__(m)

        try:
            self_read_fn(self._filename)
            self._cplex = cpx
            return cpx

        except CplexError as cpx_e:  # pragma: no cover
            # delete cplex instance
            del cpx
            raise ModelReaderError("*CPLEX error {0!s} reading file {1} - exiting".format(cpx_e, self._filename))

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        cpx = self._cplex
        if cpx is not None:
            del cpx
            self._cplex = None


class ModelReader(object):
    """ This class is used to read models from CPLEX files.

    All keyword arguments understood by the `Model` class can be passed to the `ModelReader` constructor
    in which case, these arguments will be used to create the initial empty model.

    Args:
        model_class: A subclass of `Model` (the default). This class type
            is used to build the empty model and fill it with the contents of the file.
            For example, to build an instance of `AdvModel`, pass `model_class=AdvModel`
            to the constructor of `ModelReader`.

    Returns:
        An instance of :class:`doc.mp.model.Model` if the file was successfully read, or None.

    Note:
        This class requires CPLEX to be installed and present in ``PYTHONPATH``. The following file formats are
        accepted: LP, SAV, MPS.

    Example:
        Reads the contents of file ``mymodel.sav`` into an `AdvModel` instance, built with the context `my_ctx`::

            mr = ModelReader(model_class=AdvModel)
            mr.read_model('mymodel.sav', context=my_ctx)

    """

    class _RangeData(object):
        # INTERNAL
        def __init__(self, var_index, var_name, lb=0, ub=1e+75):
            self.var_index = var_index
            self.var_name = var_name
            self.lb = lb
            self.ub = ub

    @staticmethod
    def _build_linear_expr_from_sparse_pair(lfactory, var_map, cpx_sparsepair):
        expr = lfactory.linear_expr(arg=0, safe=True)
        for ix, k in izip(cpx_sparsepair.ind, cpx_sparsepair.val):
            dv = var_map[ix]
            expr._add_term(dv, k)
        return expr

    def __init__(self, **kwargs):
        pass

    @classmethod
    def read(cls, pathname, model_name=None, verbose=False, model_class=None, **kwargs):
        """
        A class method to read a model from a file.

        :param pathname: a path to the file to read
        :param model_name: An optional string to use as name for the returned model.
            If None, the basename of the path is used.
        :param verbose: An optional flag to print informative messages, default is False.
        :param model_class: An optional class type; must be a subclass of Model.
            The returned model is built using this model_class and the keyword arguments kwargs, if any.
            By default, the model is class is `Model`.

        kwargs: A dict of keyword-based arguments that are used when creating the model
            instance.

        :return: a model instance, or None, if an error occurred.

        Note:
            This class method calls `ModelReader.read_model()`, without requesting to create an explicit instance
            of `ModelReader`.
        """
        mri = ModelReader()
        return mri.read_model(pathname, model_name, verbose, model_class, **kwargs)

    @classmethod
    def read_prm(cls, filename):
        """ Reads a CPLEX PRM file.

        Reads a CPLEX parameters file and returns a DOcplex parameter group
        instance. This parameter object can be used in a solve().

        Args:
            filename: a path string

        Returns:
            A `RootParameterGroup object`, if the read operation succeeds, else None.
        """
        if not Cplex:  # pragma: no cover
            raise RuntimeError("ModelReader.read_prm() requires CPLEX runtime.")
        with _CplexReaderFileContext(filename, read_method=["parameters", "read_file"]) as cpx:
            if cpx:
                # raw parameters
                params = get_params_from_cplex_version(cpx.get_version())
                for param in params:
                    try:
                        cpx_value = cpx._env.parameters._get(param.cpx_id)
                        if cpx_value != param.default_value:
                            param.set(cpx_value)

                    except CplexError:  # pragma: no cover
                        pass
                return params
            else:
                return None

    @staticmethod
    def _safe_call_get_names(interface, fallback_names=None):
        # cplex crashes when calling get_names on some files (e.g. SAV)
        # in this case filter out error 1219
        # and return a fallback list with None or ""
        try:
            names = interface.get_names()
            return names
        except CplexSolverError as cpxse:  # pragma: no cover
            errcode = cpxse.args[2]
            # when all indicators have no names, cplex raises this error
            # CPLEX Error  1219: No names exist.
            if errcode == 1219:
                return fallback_names or []
            else:
                # this is something else
                raise

    @staticmethod
    def _cplex_read(filename, verbose=False):
        # print("-> start reading file: {0}".format(filename))
        cpx = _safe_cplex()
        # no warnings
        if not verbose:
            cpx.set_results_stream(None)
            cpx.set_log_stream(None)
            cpx.set_warning_stream(None)
            cpx.set_error_stream(None)  # remove messages about names
        try:
            cpx.read(filename)
            return cpx
        except CplexError as cpx_e:
            raise ModelReaderError("*CPLEX error {0!s} reading file {1} - exiting".format(cpx_e, filename))

    def read_model(self, filename, model_name=None, verbose=False, model_class=None, **kwargs):
        """ Reads a model from a CPLEX export file.

        Accepts all formats exported by CPLEX: LP, SAV, MPS.

        If an error occurs while reading the file, the message of the exception
        is printed and the function returns None.

        Args:
            filename: The file to read.
            model_name: An optional name for the newly created model. If None,
                the model name will be the path basename.
            verbose: An optional flag to print informative messages, default is False.
            model_class: An optional class type; must be a subclass of Model.
                The returned model is built using this model_class and the keyword arguments kwargs, if any.
                By default, the model is class is `Model` (see
            kwargs: A dict of keyword-based arguments that are used when creating the model
                instance.

        Example:
            `m = read_model("c:/temp/foo.mps", model_name="docplex_foo", solver_agent="docloud", output_level=100)`

        Returns:
            An instance of Model, or None if an exception is raised.

        See Also:
            :class:`docplex.mp.model.Model`

        """
        if not Cplex:  # pragma: no cover
            raise RuntimeError("ModelReader.read_model() requires CPLEX runtime.")

        if not os.path.exists(filename):
            raise IOError("* file not found: {0}".format(filename))

        # extract basename
        if model_name:
            name_to_use = model_name
        else:
            basename = os.path.basename(filename)
            dotpos = basename.find(".")
            if dotpos > 0:
                name_to_use = basename[:dotpos]
            else:
                name_to_use = basename

        model_class = model_class or Model

        if 0 == os.stat(filename).st_size:
            print("* file is empty: {0} - exiting".format(filename))
            return model_class(name=name_to_use, **kwargs)

        # print("-> start reading file: {0}".format(filename))
        cpx = self._cplex_read(filename, verbose=verbose)

        if not cpx:  # pragma: no cover
            return None

        range_map = {}
        final_output_level = kwargs.get("output_level", "info")
        debug_read = kwargs.get("debug", False)

        try:
            # force no tck
            if 'checker' in kwargs:
                final_checker = kwargs['checker']
            else:
                final_checker = 'default'
            # build the model with no checker, then restore final_checker in the end.
            kwargs['checker'] = 'off'
            # -------------

            mdl = model_class(name=name_to_use, **kwargs)
            lfactory = mdl._lfactory
            qfactory = mdl._qfactory
            mdl.set_quiet()  # output level set to ERROR
            vartype_cont = mdl.continuous_vartype
            vartype_map = {'B': mdl.binary_vartype,
                           'I': mdl.integer_vartype,
                           'C': mdl.continuous_vartype,
                           'S': mdl.semicontinuous_vartype}
            # 1 upload variables
            cpx_nb_vars = cpx.variables.get_num()
            cpx_var_names = self._safe_call_get_names(cpx.variables)

            if cpx._is_MIP():
                cpx_vartypes = [vartype_map.get(cpxt, vartype_cont) for cpxt in cpx.variables.get_types()]
            else:
                cpx_vartypes = [vartype_cont] * cpx_nb_vars
            cpx_var_lbs = cpx.variables.get_lower_bounds()
            cpx_var_ubs = cpx.variables.get_upper_bounds()
            # map from cplex variable indices to docplex's
            # use to skip range vars
            # cplex : [x, Rg1, y] -> {0:0, 2: 1}
            var_index_map = {}

            d = 0
            model_varnames = []
            model_lbs = []
            model_ubs = []
            model_types = []
            for v in range(cpx_nb_vars):
                varname = cpx_var_names[v] if cpx_var_names else None

                if varname and varname.startswith("Rg"):
                    # generated var for ranges
                    range_map[v] = self._RangeData(var_index=v, var_name=varname, ub=cpx_var_ubs[v])
                else:
                    # docplex_var = lfactory.new_var(vartype, lb, ub, varname)
                    var_index_map[v] = d
                    model_varnames.append(varname)
                    model_types.append(cpx_vartypes[v])
                    model_lbs.append(cpx_var_lbs[v])
                    model_ubs.append(cpx_var_ubs[v])
                    d += 1

            # vars
            model_vars = lfactory.new_multitype_var_list(d,
                                                         model_types,
                                                         model_lbs,
                                                         model_ubs,
                                                         model_varnames)

            cpx_var_index_to_docplex = {v: model_vars[var_index_map[v]] for v in var_index_map.keys()}

            # 2. upload linear constraints and ranges (mixed in cplex)
            cpx_linearcts = cpx.linear_constraints
            nb_linear_cts = cpx_linearcts.get_num()
            all_rows = cpx_linearcts.get_rows()
            all_rhs = cpx_linearcts.get_rhs()
            all_senses = cpx_linearcts.get_senses()
            all_range_values = cpx_linearcts.get_range_values()
            cpx_ctnames = self._safe_call_get_names(cpx_linearcts)

            has_range = range_map or any(s == "R" for s in all_senses)
            deferred_cts = []

            for c in range(nb_linear_cts):
                row = all_rows[c]
                sense = all_senses[c]
                rhs = all_rhs[c]
                ctname = cpx_ctnames[c] if cpx_ctnames else None
                range_val = all_range_values[c]

                indices = row.ind
                coefs = row.val
                range_data = None

                if not has_range:
                    expr = mdl._aggregator._scal_prod((cpx_var_index_to_docplex[idx] for idx in indices), coefs)
                    op = ComparisonType.parse(sense)
                    ct = lfactory._new_binary_constraint(lhs=expr, rhs=rhs, sense=op)
                    ct.name = ctname
                    deferred_cts.append(ct)

                else:
                    expr = lfactory.linear_expr()
                    rcoef = 1
                    for idx, koef in izip(indices, coefs):
                        var = cpx_var_index_to_docplex.get(idx, None)
                        if var:
                            expr._add_term(var, koef)
                        elif idx in range_map:
                            # this is a range: coeff must be 1 or -1
                            abscoef = koef if koef >= 0 else -koef
                            rcoef = koef
                            assert abscoef == 1, "range var has coef different from 1: {}".format(koef)
                            assert range_data is None, "range_data is not None: {0!s}".format(
                                range_data)  # cannot use two range vars
                            range_data = range_map[idx]
                        else:  # pragma: no cover
                            # this is an internal error.
                            raise ModelReaderError("ERROR: index not in var map or range map: {0}".format(idx))

                    if range_data:
                        label = ctname or 'c#%d' % (c + 1)
                        if sense not in "EL":  # pragma: no cover
                            raise ModelReaderError("{0} range sense is not E: {1!s}".format(label, sense))
                        if rcoef < 0:  # -1 actually
                            rng_lb = rhs
                            rng_ub = rhs + range_data.ub
                        elif rcoef > 0:  # koef is 1 here
                            rng_lb = rhs - range_data.ub
                            rng_ub = rhs
                        else:  # pragma: no cover
                            raise ModelReaderError("unexpected range coef: {}".format(rcoef))

                        mdl.add_range(lb=rng_lb, expr=expr, ub=rng_ub, rng_name=ctname)
                    else:
                        if sense == 'R':
                            # range min is rangeval
                            range_lb = rhs
                            range_ub = rhs + range_val
                            mdl.add_range(lb=range_lb, ub=range_ub, expr=expr, rng_name=ctname)
                        else:
                            op = ComparisonType.cplex_ctsense_to_python_op(sense)
                            ct = op(expr, rhs)
                            mdl.add_constraint(ct, ctname)
            if deferred_cts:
                # add constraint as a block
                lfactory._post_constraint_block(posted_cts=deferred_cts)

            # 3. upload Quadratic constraints
            cpx_quadraticcts = cpx.quadratic_constraints
            nb_quadratic_cts = cpx_quadraticcts.get_num()
            all_rhs = cpx_quadraticcts.get_rhs()
            all_linear_nb_non_zeros = cpx_quadraticcts.get_linear_num_nonzeros()
            all_linear_components = cpx_quadraticcts.get_linear_components()
            all_quadratic_nb_non_zeros = cpx_quadraticcts.get_quad_num_nonzeros()
            all_quadratic_components = cpx_quadraticcts.get_quadratic_components()
            all_senses = cpx_quadraticcts.get_senses()
            cpx_ctnames = self._safe_call_get_names(cpx_quadraticcts)

            for c in range(nb_quadratic_cts):
                rhs = all_rhs[c]
                linear_nb_non_zeros = all_linear_nb_non_zeros[c]
                linear_component = all_linear_components[c]
                quadratic_nb_non_zeros = all_quadratic_nb_non_zeros[c]
                quadratic_component = all_quadratic_components[c]
                sense = all_senses[c]
                ctname = cpx_ctnames[c] if cpx_ctnames else None

                if linear_nb_non_zeros > 0:
                    indices, coefs = linear_component.unpack()
                    linexpr = mdl._aggregator._scal_prod((cpx_var_index_to_docplex[idx] for idx in indices), coefs)
                else:
                    linexpr = None

                if quadratic_nb_non_zeros > 0:
                    qfactory = mdl._qfactory
                    ind1, ind2, coefs = quadratic_component.unpack()
                    quads = qfactory.term_dict_type()
                    for idx1, idx2, coef in izip(ind1, ind2, coefs):
                        quads[VarPair(cpx_var_index_to_docplex[idx1], cpx_var_index_to_docplex[idx2])] = coef

                else:  # pragma: no cover
                    # should not happen, but who knows
                    quads = None

                quad_expr = mdl._aggregator._quad_factory.new_quad(quads=quads, linexpr=linexpr, safe=True)
                op = ComparisonType.cplex_ctsense_to_python_op(sense)
                ct = op(quad_expr, rhs)
                mdl.add_constraint(ct, ctname)

            # 4. upload indicators
            cpx_indicators = cpx.indicator_constraints
            nb_indicators = cpx_indicators.get_num()
            all_ind_names = self._safe_call_get_names(cpx_indicators)

            all_ind_bvars = cpx_indicators.get_indicator_variables()
            all_ind_rhs = cpx_indicators.get_rhs()
            all_ind_linearcts = cpx_indicators.get_linear_components()
            all_ind_senses = cpx_indicators.get_senses()
            all_ind_complemented = cpx_indicators.get_complemented()
            lfactory = mdl._lfactory
            for i in range(nb_indicators):
                ind_bvar = all_ind_bvars[i]
                ind_name = all_ind_names[i] if all_ind_names else None
                ind_rhs = all_ind_rhs[i]
                ind_linear = all_ind_linearcts[i]  # SparsePair(ind, val)
                ind_sense = all_ind_senses[i]
                ind_complemented = all_ind_complemented[i]
                # 1 . check the bvar is ok
                ind_bvar = cpx_var_index_to_docplex[ind_bvar]
                # each var appears once
                ind_linexpr = self._build_linear_expr_from_sparse_pair(lfactory, cpx_var_index_to_docplex, ind_linear)
                op = ComparisonType.cplex_ctsense_to_python_op(ind_sense)
                ind_ct = op(ind_linexpr, ind_rhs)
                indct = lfactory.new_indicator_constraint(ind_bvar, ind_ct,
                                                          active_value=1 - ind_complemented, name=ind_name)
                mdl.add(indct)

            # 5. upload Piecewise linear constraints
            try:
                cpx_pwl = cpx.pwl_constraints
                cpx_pwl_defs = cpx_pwl.get_definitions()
                pwl_fallback_names = [""] * cpx_pwl.get_num()
                cpx_pwl_names = self._safe_call_get_names(cpx_pwl, pwl_fallback_names)
                for (vary_idx, varx_idx, preslope, postslope, breakx, breaky), pwl_name in izip(cpx_pwl_defs,
                                                                                                cpx_pwl_names):
                    varx = cpx_var_index_to_docplex.get(varx_idx, None)
                    vary = cpx_var_index_to_docplex.get(vary_idx, None)
                    breakxy = [(brkx, brky) for brkx, brky in zip(breakx, breaky)]
                    pwl_func = mdl.piecewise(preslope, breakxy, postslope, name=pwl_name)
                    pwl_expr = mdl._lfactory.new_pwl_expr(pwl_func, varx, 0, add_counter_suffix=False, resolve=False)
                    pwl_expr._f_var = vary
                    pwl_expr._ensure_resolved()

            except AttributeError:  # pragma: no cover
                pass  # Do not check for PWLs if Cplex version does not support them

            # 6. upload objective
            cpx_obj = cpx.objective
            cpx_sense = cpx_obj.get_sense()

            cpx_all_lin_obj_coeffs = cpx_obj.get_linear()
            # noinspection PyPep8
            all_obj_vars = []
            all_obj_coefs = []

            for v in range(cpx_nb_vars):
                if v in cpx_var_index_to_docplex:
                    obj_coeff = cpx_all_lin_obj_coeffs[v]
                    all_obj_coefs.append(obj_coeff)
                    all_obj_vars.append(cpx_var_index_to_docplex[v])
                    #  obj_expr._add_term(idx_to_var_map[v], cpx_all_obj_coeffs[v])
            obj_expr = mdl._aggregator._scal_prod(all_obj_vars, all_obj_coefs)

            if cpx_obj.get_num_quadratic_variables() > 0:
                cpx_all_quad_cols_coeffs = cpx_obj.get_quadratic()
                quads = qfactory.term_dict_type()
                for v, col_coefs in izip(cpx_var_index_to_docplex, cpx_all_quad_cols_coeffs):
                    var1 = cpx_var_index_to_docplex[v]
                    indices, coefs = col_coefs.unpack()
                    for idx, coef in izip(indices, coefs):
                        vp = VarPair(var1, cpx_var_index_to_docplex[idx])
                        quads[vp] = quads.get(vp, 0) + coef / 2

                obj_expr += qfactory.new_quad(quads=quads, linexpr=None)

            obj_expr += cpx.objective.get_offset()
            is_maximize = cpx_sense == ObjSense.maximize

            if is_maximize:
                mdl.maximize(obj_expr)
            else:
                mdl.minimize(obj_expr)

            # upload sos
            cpx_sos = cpx.SOS
            cpx_sos_num = cpx_sos.get_num()
            if cpx_sos_num > 0:
                cpx_sos_types = cpx_sos.get_types()
                cpx_sos_indices = cpx_sos.get_sets()
                cpx_sos_names = cpx_sos.get_names()
                if not cpx_sos_names:
                    cpx_sos_names = [None] * cpx_sos_num
                for sostype, sos_sparse, sos_name in izip(cpx_sos_types, cpx_sos_indices, cpx_sos_names):
                    sos_var_indices = sos_sparse.ind
                    isostype = int(sostype)
                    sos_vars = [cpx_var_index_to_docplex[var_ix] for var_ix in sos_var_indices]
                    mdl.add_sos(dvars=sos_vars, sos_arg=isostype, name=sos_name)

            # upload lazy constraints
            cpx_linear_advanced = cpx.linear_constraints.advanced
            cpx_lazyct_num = cpx_linear_advanced.get_num_lazy_constraints()
            if cpx_lazyct_num:
                print("WARNING: found {0} lazy constraints that cannot be uploaded to DOcplex".format(cpx_lazyct_num))

            mdl.output_level = final_output_level
            if final_checker:
                # need to restore checker
                mdl.set_checker(final_checker)

        except CplexError as cpx_e:  # pragma: no cover
            print("* CPLEX error: {0!s} reading file {1}".format(cpx_e, filename))
            mdl = None
            if debug_read:
                raise

        except ModelReaderError as mre:  # pragma: no cover
            print("! Model reader error: {0!s} while reading file {1}".format(mre, filename))
            mdl = None
            if debug_read:
                raise

        except DOcplexException as doe:  # pragma: no cover
            print("! Internal DOcplex error: {0!s} while reading file {1}".format(doe, filename))
            mdl = None
            if debug_read:
                raise

        except Exception as any_e:  # pragma: no cover
            print("Internal exception raised: {0!s} while reading file {1}".format(any_e, filename))
            mdl = None
            if debug_read:
                raise

        finally:
            # clean up CPLEX instance...
            del cpx

        return mdl
