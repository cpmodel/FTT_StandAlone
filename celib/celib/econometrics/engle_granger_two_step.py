# -*- coding: utf-8 -*-
"""
engle_granger_two_step
======================
Functions to perform automated general-to-specific (Gets) estimation of
single-equation error-correction models using the Engle-Granger two-step
procedure.

*** WARNING: This module is experimental and not maintained.
             It is also undocumented and largely untested.
             We provide the code as-is and you use it at your own risk. ***

TODO:
 - add documentation
 - add diagnostics tools (dump estimation df, params, se, ... into an Excel spreadsheet)
 - performance?

"""

import warnings
warnings.warn('This module is experimental and not maintained. '
              'It is also undocumented and largely untested. '
              'We provide the code as-is and you use it at your own risk.')


import numpy as np
import pandas as pd
import re
import itertools # for combinatorics


def parse_trans(var):
    """Parse transformations like 'log(d.Y)' -> ['log', 'd', 'Y']"""
    if var[:4].lower() == "log(" and var[len(var)-1] == ")":
        return ["log"] + parse_trans(var[4:len(var)-1])

    if var[:4].lower() == "exp(" and var[len(var)-1] == ")":
        return ["exp"] + parse_trans(var[4:len(var)-1])

    if var[:2].lower() == "d.":
        return ["d"] + parse_trans(var[2:])

    if var[:2].lower() == "l.":
        return ["l"] + parse_trans(var[2:])

    return [var]


def parse_eq(eq):
    """Parse equation syntax, return list of variables and their transformations"""
    if not len(eq.split('=')) == 2:
        raise ValueError('Equation has to have exactly one equation sign.')

    dep, indep = eq.split('=')
    indep = indep.split('+')

    vars = [] # variables and transformations
    rsts = [] # restrictions
    for v in [dep] + indep:
        # first let's resolve restrictions
        m = re.match('^([a-zA-Z0-9_\.\(\)\/]+)\s*(?:\[(.*),(.*)\])?$', v.strip())

        if m == None:
            raise ValueError("Unexpected variable name syntax, '%s' " % v)

        nm = m.group(1)
        if m.group(2) == None: # no restrictions set
            rst = [-np.inf, np.inf]
        else: # set to infty if empty, otherwise just float
            lb = -np.inf if len(m.group(2)) == 0 else float(m.group(2))
            ub =  np.inf if len(m.group(3)) == 0 else float(m.group(3))
            rst = [lb, ub]

        vars.append(parse_trans(nm))
        rsts.append(rst)

    # don't return restrictions on the dependent variable, that makes little sense
    return { 'variables': vars, 'restrictions': rsts[1:] }

def variable_names(eq):
    """
    Extract variable names from an equation, e.g. 'log(Y) = d.log(C) + RPDI' -> [ 'Y', 'C', 'RPDI' ]

    Useful for preparing a DataFrame for estimation or just having an idea what the equation is about without the syntactic sugar.
    """
    ret = []

    # parse_eq() returns all the variables with the transformations (logs, diffs)
    # we take just the last element, the variable name
    for v in parse_eq(eq)["variables"]:
        ret.append(v[-1])

    return ret # TODO: filter duplicates

def transform_series_old(data, trans):
    # TODO: remove

    for t in trans[::-1]:
        if t == 'log': # log()
            data = np.log(data)
        elif t == 'exp': # exp()
            data = np.exp(data)
        elif t == 'd': # first difference
            data = np.append([np.nan], np.diff(data)) # OPTIM: have an np.empty and place stuff in it
        elif t == 'l': # lag, add NaN in the beginning
            data = np.append([np.nan], data[:-1]) # OPTIM: have something like pd.shift (scipy has this as well)
        else:
            raise ValueError("Unrecognised transformation, " + t)

    return data

def transform_series(data, trans):
    """
    Take a vector of data and apply transformations for the use in linear regressions.

    'trans' will be a result of parse_trans() or parse_eq(), that extract syntax from equations. E.g. 'log(d.Y)' -> [ 'log', 'd', 'Y' ]. It then, if supplied Y data, takes the first difference and then logs them.
    """
    # TODO: sort out this typecheck
    # if not isinstance(data, np.Series):
    # raise ValueError("Data need to be supplied as a numpy series")
    for t in trans[::-1]: # start with the most inner transformation
        if t == 'log': # log()
            data = np.log(data)
        elif t == 'exp': # exp()
            data = np.exp(data)
        elif t == 'd': # first difference
            ret = np.empty(len(data))
            ret[0] = np.nan
            ret[1:] = np.diff(data)
            data = ret
        elif t == 'l': # lag, add NaN in the beginning
            ret = np.empty(len(data))
            ret[0] = np.nan
            ret[1:] = data[:-1]
            data = ret
        else:
            raise ValueError("Unrecognised transformation, %s" % t)

    return data

def linreg(x,y):
    """Return OLS estimates of a linear fit, no other info provided."""
    return ( np.linalg.inv(x.transpose().dot(x))).dot( (x.transpose()).dot(y) )

def estimate(data, eq):
    """
    Estimate OLS from given data and equation specification. Data are in levels, equation specification can include transformations, e.g. "d.log(A) = log(C) + d.D + d.l.E + F" Constant is not included in the specification, but is estimated.

    Output is a dictionary with parameters (beta), Akaike information (AIC), R-squared (Rsq), and residuals (resid)
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data need to be supplied as a dataframe")

    pp = parse_eq(eq)
    vars = pp["variables"]
    restr = pp["restrictions"]
    mrestr = np.array(restr) # as two columns, will be handy in testing restrictions
    brestr = [] # reformatted restrictions; not numerical, but as sets of 0, 1, 2

    for r in restr:
        c = np.abs(r) < np.inf # how do the restrictions compare to infty?

        if all(c): # both reals, can restrict both
            rst = [0, 1, 2]
        elif not any(c): # both infties, no restrictions
            rst = [0]
        elif c[0]: # real lower bound
            rst = [0,1]
        elif c[1]: # real upper bound
            rst = [0,2]

        brestr.append(rst)

    # Combinations of restrictions, we will loop through these once we have data ready
    rr = np.array(list(itertools.product(*brestr))) # TODO: this can get very large, optimise
    rr = rr[np.sum(rr==0,1).argsort(axis=0)[::-1]] # sort by number of zeroes in the restrictions


    # Get data now, prepare an array for 'y' and a matrix for 'x'
    # Dependent variable
    ynm = vars[0].pop()

    y = transform_series(data.values[:,data.columns.get_loc(ynm)], vars[0])

    # Independent variables
    x = np.empty(shape=(len(y), len(vars)-1), dtype=float) # empty matrix to be populated

    j=0
    for v in vars[1:]:
        nm = v.pop()
        x[:,j] = transform_series(data.values[:,data.columns.get_loc(nm)], v)

        #print transform_series(data.values[:,data.columns.get_loc(nm)], v)
        j += 1

    # prepend a vector of ones (for the intercept)
    x = np.insert(x, 0, np.ones(len(x)), axis = 1)

    # find the first row with no NaNs
    for j in range(0,len(x)):
        if not ( np.any(np.isnan(x[j,:])) or np.isnan(y[j]) or np.any(np.isinf(x[j,:])) or np.isinf(y[j]) ):
            x = x[j:,:]
            y = y[j:]
            break
        elif j == len(x)-1:
           #print data
           raise ValueError("All observations are invalid in this equation.")

    # Data ready
    # loop through possible restriction combinations and try them
    buffer = None # to store successful solutions
    for j in range(0, len(rr)): # not iterating so that we can retain a lookahead
        r = rr[j]
        xvar = x.copy()
        yvar = y.copy()

        #print 'Trying %s' % r

        # We first need to prep data in case there are restrictions
        if not all(r == 0):
            for k in range(0, len(r))[::-1]: # go from the last one, to remove the right columns
                if r[k]==0: continue # nothing to do, no restriction
                yvar -= mrestr[k, r[k]-1] * xvar[:,k+1] # subtract x[j] from y
                xvar = np.delete(xvar, k+1, axis=1) # remove relevant column from X


        beta = linreg(xvar, yvar)
        beta_restr = np.copy(beta) # save for fitting purposes

        # place constant parameters back in
        if not all(r == 0):
            for k in range(0, len(r)): # this time go forward
                if r[k]==0: continue # nothing to do, no restriction
                beta = np.insert(beta, k+1, mrestr[k, r[k]-1])

        # All restrictions satisfied, save solution
        if all(beta[1:] <= mrestr[:,1]) and all(beta[1:] >= mrestr[:,0]):

            # import pickle
            # pickle.dump({'beta': beta, 'xvar': xvar, 'yvar': yvar}, open('est.pk', 'wb'))
            resid = yvar - np.sum(beta_restr * xvar, axis=1)
            varmat = np.var(resid) * np.linalg.inv(xvar.transpose().dot(xvar))

            # calculate standard errors
            # TODO: calculate standard error of slope!
            se = [np.nan] + [np.sqrt(varmat[j,j]) for j in range(1,np.shape(varmat)[0])]

            # put NaN errors for restricted variables as they were not calculated
            if not all(r == 0):
                for k in range(0, len(r)):
                    if r[k]==0: continue # nothing to do, no restriction
                    se = np.insert(se, k+1, np.nan)



            # TODO: test if sum(resid) == 0!!!
            RSS = np.sum(map(lambda x: x**2, resid))
            TSS = sum(map(lambda x: x**2, np.mean(y) - y))

            k = np.shape(x)[1] + 1
            n = len(y)

            Rsq = 1 - RSS/TSS
            AIC = n * np.log(RSS/n) + 2*k

            # Only save solution is if it's
            #   a) the first one, or
            #   b) better than the existing (AIC higher)
            if buffer == None or buffer["AIC"] > AIC:
                buffer = {
                    "AIC": AIC,
                    "Rsq": Rsq,
                    "beta": beta,
                    "resid": resid,
                    "se": se
                }


        # Are we moving onto more restrictions? If so, check the buffer.
        if sum(rr[min(j+1, len(rr)-1)] > 0) > sum(r > 0):
            if buffer == None: # there are no solutions yet
                continue
            else: # solutions found, we can exit out of the for loop
                break
        # finished: for loop through restrictions

    return buffer

def egranger(data, spec):
    lr = estimate(data, spec["LR"])
    if lr == None:
        raise ValueError('Failed to estimate the long-run equation for: %s' % spec["LR"])

    sdata = data.copy()

    nr = len(lr["resid"])
    n = len(data)

    lre = np.empty(n)
    lre.fill(np.nan)
    lre[n-nr:] = lr["resid"]

    sdata['LRerr'] = lre

    sr = estimate(sdata, spec["SR"])

    return { "SR" : sr, "LR": lr }

def fit_eq(data, spec, params):
    # OPTIM: so much to do here
    '''Fit equation using the last row of data. 'Spec' is a string, to be parsed using `parse_eq`'''

    pr = list(params) # convert numpy array to a list (so we can pop)
    vars = parse_eq(spec)["variables"]
    rhs = pr.pop(0) # right hand side starts with the intercept

    # form right hand side values
    for v in vars[1:]: # all except the dependent variable
        vname = v.pop() # last item in the array is the variable name
        ser = transform_series(data.values[:,data.columns.get_loc(vname)], v) # this is our transformed data now
        # take last datapoint in the data series and multiply with the first parameter (and remove it)
        rhs += ser[-1] * pr.pop(0)

    # now that the right hand side is sorted, we need to figure out what happens
    # to the left hand side
    # if log(Y) = ..., then Y = exp(rhs); if d.Y = ..., then Y = rhs + L.Y etc.
    dvn = vars[0].pop() # name of dependent variable
    tr = vars[0] # remaining transformations of it
    dd = data.values[:,data.columns.get_loc(dvn)] # series of levels of our dependent variable

    # test various possibilities of what needs to be returned
    if tr == ['log']: # log(dep) = rhs  =>  dep = exp(rhs)
        fit_level = np.exp(rhs)
    elif tr == ['d']: # d.dep = rhs => dep = L.dep + rhs
        fit_level = dd[-2] + rhs
    elif tr == ['d', 'log']: # d.log(dep) = rsh  =>  dep = exp(log(L.dep) + RHS)
        fit_level = np.exp(np.log(dd[-2]) + rhs)
    else:
        raise ValueError("Unrecognised transformation of dependent variable: " + (', ').join(tr))

    # return original right hand side and then the level
    # also return the actuals as `data`
    ac = dd[-1] # actuals
    trac = transform_series(dd, tr) # transformed series

    # if the dependent variable is a fraction, we need to premultiply the result by the denominator
    if '/' in dvn:
        num, denom = dvn.split('/')
        denom_value = data.values[-1, data.columns.get_loc(denom)] # last *actual* value of the denominator
        fit_level *= denom_value
        ac        *= denom_value # divide the actuals as well

    return {"fit": rhs, "fit_level": fit_level, "actual_level": ac, "actual": trac[-1]}

def fit_egranger(data, spec, params, inplace=False):
    """
        TODO: add documentation
    """
    lrf = fit_eq(data[:-1], spec['LR'], params['LR']) # fit it using data until last year

    # if we don't care what happens to the dataframe passed, let it be modified (it's just the LR error bit)
    if inplace:
        dd = data
    else:
        dd = data.copy() # OPTIM: do we mind the source data are being amended?

    dd['LRerr'] = np.nan
    dd.values[-2,dd.columns.get_loc('LRerr')] = lrf['actual'] - lrf['fit'] # set residual from the LR equation

    return fit_eq(dd, spec['SR'], params['SR'])
