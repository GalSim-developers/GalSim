# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
import galsim


def _GenerateFromRandom(config, base, value_type):
    """@brief Return a random value drawn from a uniform distribution
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for Random")
    rng = base['rng']
    ud = galsim.UniformDeviate(rng)

    # Each value_type works a bit differently:
    if value_type is galsim.Angle:
        import math
        galsim.config.CheckAllParams(config)
        val = ud() * 2 * math.pi * galsim.radians
        #print base['obj_num'],'Random angle = ',val
        return val, False
    elif value_type is bool:
        galsim.config.CheckAllParams(config)
        val = ud() < 0.5
        #print base['obj_num'],'Random bool = ',val
        return val, False
    else:
        ignore = [ 'default' ]
        req = { 'min' : value_type , 'max' : value_type }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, ignore=ignore)

        min = kwargs['min']
        max = kwargs['max']

        if value_type is int:
            import math
            val = int(math.floor(ud() * (max-min+1))) + min
            # In case ud() == 1
            if val > max: val = max
        else:
            val = ud() * (max-min) + min

        #print base['obj_num'],'Random = ',val
        return val, False


def _GenerateFromRandomGaussian(config, base, value_type):
    """@brief Return a random value drawn from a Gaussian distribution
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for RandomGaussian")
    rng = base['rng']

    req = { 'sigma' : float }
    opt = { 'mean' : float, 'min' : float, 'max' : float }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)

    sigma = kwargs['sigma']

    if 'gd' in base and base['current_gdsigma'] == sigma:
        # Minor subtlety here.  GaussianDeviate requires two random numbers to 
        # generate a single Gaussian deviate.  But then it gets a second 
        # deviate for free.  So it's more efficient to store gd than to make
        # a new one each time.  So check if we did that.
        gd = base['gd']
    else:
        # Otherwise, just go ahead and make a new one.
        gd = galsim.GaussianDeviate(rng,sigma=sigma)
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
        if min == mean:
            do_abs = True
            max -= mean
            min = -max
        elif max == mean:
            do_abs = True
            do_neg = True
            min -= mean
            max = -min
        else:
            min -= mean
            max -= mean
    
        # Emulate a do-while loop
        import math
        while True:
            val = gd()
            if do_abs: val = math.fabs(val)
            if val >= min and val <= max: break
        if do_neg: val = -val
        val += mean
    else:
        val = gd()
        if 'mean' in kwargs: val += kwargs['mean']

    #print base['obj_num'],'RandomGaussian: ',val
    return val, False

def _GenerateFromRandomPoisson(config, base, value_type):
    """@brief Return a random value drawn from a Poisson distribution
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for RandomPoisson")
    rng = base['rng']

    req = { 'mean' : float }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req)

    mean = kwargs['mean']

    dev = galsim.PoissonDeviate(rng,mean=mean)
    val = dev()

    #print base['obj_num'],'RandomPoisson: ',val
    return val, False

def _GenerateFromRandomBinomial(config, base, value_type):
    """@brief Return a random value drawn from a Binomial distribution
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for RandomBinomial")
    rng = base['rng']

    req = {}
    opt = { 'p' : float }

    # Let N be optional for bool, since N=1 is the only value that makes sense.
    if value_type is bool:
        opt['N'] = int
    else:
        req['N'] = int
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)

    N = kwargs.get('N',1)
    p = kwargs.get('p',0.5)
    if value_type is bool and N != 1:
        raise ValueError("N must = 1 for RandomBinomial used in bool context")

    dev = galsim.BinomialDeviate(rng,N=N,p=p)
    val = dev()

    #print base['obj_num'],'RandomBinomial: ',val
    return val, False


def _GenerateFromRandomWeibull(config, base, value_type):
    """@brief Return a random value drawn from a Weibull distribution
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for RandomWeibull")
    rng = base['rng']

    req = { 'a' : float, 'b' : float }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req)

    a = kwargs['a']
    b = kwargs['b']
    dev = galsim.WeibullDeviate(rng,a=a,b=b)
    val = dev()

    #print base['obj_num'],'RandomWeibull: ',val
    return val, False


def _GenerateFromRandomGamma(config, base, value_type):
    """@brief Return a random value drawn from a Gamma distribution
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for RandomGamma")
    rng = base['rng']

    req = { 'k' : float, 'theta' : float }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req)

    k = kwargs['k']
    theta = kwargs['theta']
    dev = galsim.GammaDeviate(rng,k=k,theta=theta)
    val = dev()

    #print base['obj_num'],'RandomGamma: ',val
    return val, False


def _GenerateFromRandomChi2(config, base, value_type):
    """@brief Return a random value drawn from a Chi^2 distribution
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for RandomChi2")
    rng = base['rng']

    req = { 'n' : float }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req)

    n = kwargs['n']

    dev = galsim.Chi2Deviate(rng,n=n)
    val = dev()

    #print base['obj_num'],'RandomChi2: ',val
    return val, False

def _GenerateFromRandomDistribution(config, base, value_type):
    """@brief Return a random value drawn from a user-defined probability distribution
    """
    if 'rng' not in base:
        raise ValueError("No rng available for RandomDistribution")
    rng = base['rng']

    opt = {'function' : str, 'interpolant' : str, 'npoints' : int, 
           'x_min' : float, 'x_max' : float }
    kwargs, safe = galsim.config.GetAllParams(config, base, opt=opt)
    
    if '_distdev' in config:
        # The overhead for making a DistDeviate is large enough that we'd rather not do it every 
        # time, so first check if we've already made one:
        distdev = config['_distdev']
        if config['_distdev_kwargs'] != kwargs:
            distdev=galsim.DistDeviate(rng,**kwargs)
            config['_distdev'] = distdev
            config['_distdev_kwargs'] = kwargs
    else:
        # Otherwise, just go ahead and make a new one.
        distdev=galsim.DistDeviate(rng,**kwargs)
        config['_distdev'] = distdev
        config['_distdev_kwargs'] = kwargs

    # Typically, the rng will change between successive calls to this, so reset the 
    # seed.  (The other internal calculations don't need to be redone unless the rest of the
    # kwargs have been changed.)
    distdev.reset(rng)

    val = distdev()
    #print base['obj_num'],'distdev = ',val
    return val, False


def _GenerateFromRandomCircle(config, base, value_type):
    """@brief Return a PositionD drawn from a circular top hat distribution.
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for= RandomCircle")
    rng = base['rng']

    req = { 'radius' : float }
    opt = { 'inner_radius' : float, 'center' : galsim.PositionD }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    radius = kwargs['radius']

    ud = galsim.UniformDeviate(rng)
    max_rsq = radius**2
    if 'inner_radius' in kwargs:
        inner_radius = kwargs['inner_radius']
        min_rsq = inner_radius**2
    else:
        min_rsq = 0.
    # Emulate a do-while loop
    while True:
        x = (2*ud()-1) * radius
        y = (2*ud()-1) * radius
        rsq = x**2 + y**2
        if rsq >= min_rsq and rsq <= max_rsq: break

    pos = galsim.PositionD(x,y)
    if 'center' in kwargs:
        pos += kwargs['center']

    #print base['obj_num'],'RandomCircle: ',pos
    return pos, False


