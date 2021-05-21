# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
from __future__ import print_function

from .util import GetRNG
from .value import GetAllParams, CheckAllParams, RegisterValueType
from ..errors import GalSimConfigError, GalSimConfigValueError
from ..random import UniformDeviate, GaussianDeviate, PoissonDeviate, BinomialDeviate
from ..random import WeibullDeviate, GammaDeviate, Chi2Deviate, DistDeviate
from ..angle import Angle, radians
from ..position import PositionD
from ..table import LookupTable

# This file adds extra value types involving random deviates: Random, RandomGaussian,
# RandomPoisson, RandomBinomial, RandomWeibull, RandomGamma, RandomChi2, RandomDistribution,
# and RandomCircle.

def _GenerateFromRandom(config, base, value_type):
    """Return a random value drawn from a uniform distribution
    """
    rng = GetRNG(config, base)
    ud = UniformDeviate(rng)

    # Each value_type works a bit differently:
    if value_type is Angle:
        import math
        CheckAllParams(config)
        val = ud() * 2 * math.pi * radians
        #print(base['obj_num'],'Random angle = ',val)
        return val, False
    elif value_type is bool:
        opt = { 'p' : float }
        kwargs, safe = GetAllParams(config, base, opt=opt)
        p = kwargs.get('p', 0.5)
        val = ud() < p
        #print(base['obj_num'],'Random bool = ',val)
        return val, False
    else:
        ignore = [ 'default' ]
        req = { 'min' : value_type , 'max' : value_type }
        kwargs, safe = GetAllParams(config, base, req=req, ignore=ignore)

        min = kwargs['min']
        max = kwargs['max']

        if value_type is int:
            import math
            val = int(math.floor(ud() * (max-min+1))) + min
            # In case ud() == 1
            if val > max: val = max
        else:
            val = ud() * (max-min) + min

        #print(base['obj_num'],'Random = ',val)
        return val, False


def _GenerateFromRandomGaussian(config, base, value_type):
    """Return a random value drawn from a Gaussian distribution
    """
    rng = GetRNG(config, base)

    req = { 'sigma' : float }
    opt = { 'mean' : float, 'min' : float, 'max' : float }
    kwargs, safe = GetAllParams(config, base, req=req, opt=opt)

    sigma = kwargs['sigma']

    if 'gd' in base and base['current_gdsigma'] == sigma:
        # Minor subtlety here.  GaussianDeviate requires two random numbers to
        # generate a single Gaussian deviate.  But then it gets a second
        # deviate for free.  So it's more efficient to store gd than to make
        # a new one each time.  So check if we did that.
        gd = base['gd']
    else:
        # Otherwise, just go ahead and make a new one.
        gd = GaussianDeviate(rng,sigma=sigma)
        base['gd'] = gd
        base['current_gdsigma'] = sigma

    if 'min' in kwargs or 'max' in kwargs:
        # Clip at min/max.
        # However, special cases if min == mean or max == mean
        #  -- can use fabs to double the chances of falling in the range.
        mean = kwargs.get('mean',0.)
        min = kwargs.get('min',-float('inf'))
        max = kwargs.get('max',float('inf'))

        do_abs = False
        do_neg = False
        if (min >= mean) and (max > mean):
            do_abs = True
            lo = min - mean
            hi = max - mean
        elif (min < mean) and (max <= mean):
            do_abs = True
            do_neg = True
            hi = mean - min
            lo = mean - max
        else:
            lo = min - mean
            hi = max - mean

        # Emulate a do-while loop
        import math
        while True:
            val = gd()
            if do_abs: val = math.fabs(val)
            if val >= lo and val <= hi: break
        if do_neg: val = -val
        val += mean
    else:
        val = gd()
        if 'mean' in kwargs: val += kwargs['mean']

    #print(base['obj_num'],'RandomGaussian: ',val)
    return val, False

def _GenerateFromRandomPoisson(config, base, value_type):
    """Return a random value drawn from a Poisson distribution
    """
    rng = GetRNG(config, base)

    req = { 'mean' : float }
    kwargs, safe = GetAllParams(config, base, req=req)

    mean = kwargs['mean']

    dev = PoissonDeviate(rng,mean=mean)
    val = dev()

    #print(base['obj_num'],'RandomPoisson: ',val)
    return val, False

def _GenerateFromRandomBinomial(config, base, value_type):
    """Return a random value drawn from a Binomial distribution
    """
    rng = GetRNG(config, base)

    req = {}
    opt = { 'p' : float }

    # Let N be optional for bool, since N=1 is the only value that makes sense.
    if value_type is bool:
        opt['N'] = int
    else:
        req['N'] = int
    kwargs, safe = GetAllParams(config, base, req=req, opt=opt)

    N = kwargs.get('N',1)
    p = kwargs.get('p',0.5)
    if value_type is bool and N != 1:
        raise GalSimConfigValueError(
            "N must = 1 for type = RandomBinomial used in bool context", N)

    dev = BinomialDeviate(rng,N=N,p=p)
    val = dev()

    #print(base['obj_num'],'RandomBinomial: ',val)
    return val, False


def _GenerateFromRandomWeibull(config, base, value_type):
    """Return a random value drawn from a Weibull distribution
    """
    rng = GetRNG(config, base)

    req = { 'a' : float, 'b' : float }
    kwargs, safe = GetAllParams(config, base, req=req)

    a = kwargs['a']
    b = kwargs['b']
    dev = WeibullDeviate(rng,a=a,b=b)
    val = dev()

    #print(base['obj_num'],'RandomWeibull: ',val)
    return val, False


def _GenerateFromRandomGamma(config, base, value_type):
    """Return a random value drawn from a Gamma distribution
    """
    rng = GetRNG(config, base)

    req = { 'k' : float, 'theta' : float }
    kwargs, safe = GetAllParams(config, base, req=req)

    k = kwargs['k']
    theta = kwargs['theta']
    dev = GammaDeviate(rng,k=k,theta=theta)
    val = dev()

    #print(base['obj_num'],'RandomGamma: ',val)
    return val, False


def _GenerateFromRandomChi2(config, base, value_type):
    """Return a random value drawn from a Chi^2 distribution
    """
    rng = GetRNG(config, base)

    req = { 'n' : float }
    kwargs, safe = GetAllParams(config, base, req=req)

    n = kwargs['n']

    dev = Chi2Deviate(rng,n=n)
    val = dev()

    #print(base['obj_num'],'RandomChi2: ',val)
    return val, False

def _GenerateFromRandomDistribution(config, base, value_type):
    """Return a random value drawn from a user-defined probability distribution
    """
    rng = GetRNG(config, base)

    ignore = [ 'x', 'f', 'x_log', 'f_log' ]
    opt = {'function' : str, 'interpolant' : str, 'npoints' : int,
           'x_min' : float, 'x_max' : float }
    kwargs, safe = GetAllParams(config, base, opt=opt, ignore=ignore)

    # Allow the user to give x,f instead of function to define a LookupTable.
    if 'x' in config or 'f' in config:
        if 'x' not in config or 'f' not in config:
            raise GalSimConfigError(
                "Both x and f must be provided for type=RandomDistribution")
        if 'function' in kwargs:
            raise GalSimConfigError(
                "Cannot provide function with x,f for type=RandomDistribution")
        x = config['x']
        f = config['f']
        x_log = config.get('x_log', False)
        f_log = config.get('f_log', False)
        interpolant = kwargs.pop('interpolant', 'spline')
        kwargs['function'] = LookupTable(x=x, f=f, x_log=x_log, f_log=f_log,
                                         interpolant=interpolant)
    else:
        if 'function' not in kwargs:
            raise GalSimConfigError(
                "function or x,f  must be provided for type=RandomDistribution")
        if 'x_log' in config or 'f_log' in config:
            raise GalSimConfigError(
                "x_log, f_log are invalid with function for type=RandomDistribution")

    if '_distdev' not in config or config['_distdev_kwargs'] != kwargs:
        # The overhead for making a DistDeviate is large enough that we'd rather not do it every
        # time, so first check if we've already made one:
        distdev = DistDeviate(rng,**kwargs)
        config['_distdev'] = distdev
        config['_distdev_kwargs'] = kwargs
    else:
        distdev = config['_distdev']

    # Typically, the rng will change between successive calls to this, so reset the
    # seed.  (The other internal calculations don't need to be redone unless the rest of the
    # kwargs have been changed.)
    distdev.reset(rng)

    val = distdev()
    #print(base['obj_num'],'distdev = ',val)
    return val, False


def _GenerateFromRandomCircle(config, base, value_type):
    """Return a PositionD drawn from a circular top hat distribution.
    """
    rng = GetRNG(config, base)

    req = { 'radius' : float }
    opt = { 'inner_radius' : float, 'center' : PositionD }
    kwargs, safe = GetAllParams(config, base, req=req, opt=opt)
    radius = kwargs['radius']
    inner_radius = kwargs.get('inner_radius',0)

    ud = UniformDeviate(rng)
    max_rsq = radius**2
    min_rsq = inner_radius**2

    if min_rsq >= max_rsq:
        raise GalSimConfigValueError(
            "inner_radius must be less than radius (%f) for type=RandomCircle"%(radius),
            inner_radius)

    # Emulate a do-while loop
    while True:
        x = (2*ud()-1) * radius
        y = (2*ud()-1) * radius
        rsq = x**2 + y**2
        if rsq >= min_rsq and rsq <= max_rsq: break

    pos = PositionD(x,y)
    if 'center' in kwargs:
        pos += kwargs['center']

    #print(base['obj_num'],'RandomCircle: ',pos)
    return pos, False

# Register these as valid value types
RegisterValueType('Random', _GenerateFromRandom, [ float, int, bool, Angle ])
RegisterValueType('RandomGaussian', _GenerateFromRandomGaussian, [ float ])
RegisterValueType('RandomPoisson', _GenerateFromRandomPoisson, [ float, int ])
RegisterValueType('RandomBinomial', _GenerateFromRandomBinomial, [ float, int, bool ])
RegisterValueType('RandomWeibull', _GenerateFromRandomWeibull, [ float ])
RegisterValueType('RandomGamma', _GenerateFromRandomGamma, [ float ])
RegisterValueType('RandomChi2', _GenerateFromRandomChi2, [ float ])
RegisterValueType('RandomDistribution', _GenerateFromRandomDistribution, [ float ])
RegisterValueType('RandomCircle', _GenerateFromRandomCircle, [ PositionD ])
