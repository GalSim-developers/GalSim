

class SimpleParam(object):
    """
    Descriptor that gets/sets a value, and on setting causes the GSObject's stored SBProfile to
    to be undefined for later re-initialization with the updated parameter set.

    Use it like this:

    class MyProfile(GSObject):
        flux = SimpleParam("flux")
    """

    def __init__(self, name, default=None, group="required", doc=None, update_on_set=True):
        self.name = name
        self.default = default
        self.__doc__ = doc
        if not group in ("required", "size", "optional"):
            raise TypeError("group keyword must be one of 'required', 'size' or 'optional'")
        else:
            self.group = group
        self.update_on_set = update_on_set
	
    def __get__(self, instance, cls):
        if instance is not None:
            # dict.setdefault will return the item in the dict if present, or set and return the
            # default otherwise
            return instance._data.setdefault(self.name, self.default)
        return self

    def __set__(self, instance, value):
        instance._data[self.name] = value
        if self.update_on_set:
            instance._SBProfile = None # Make sure that the ._SBProfile storage is emptied


class GetSetFuncParam(object):
    """
    Descriptor that uses user-supplied functions to get/set values, intended for defining "derived"
    quantities.

    Like SimpleParam, on setting this descriptor causes the GSObject's stored SBProfile to be
    undefined, for later re-initialization with the updated parameter set.

    Use it like this, illustrating a reworking of the GetSetScaleParam functionality to define one
    parameter as a rescaling of the other:

    class MyProfile(GSObject):
    
        half_light_radius = SimpleParam("half_light_radius")

        def _get_fwhm(self):
            return self.half_light_radius * RADIUS_CONVERSION_FACTOR
        def _set_fwhm(self, value):
            self.half_light_radius = value / RADIUS_CONVERSION_FACTOR
        fwhm = GetSetParam(_get_fwhm, _set_fwhm)

    (N.B. There is not actual need to do this, since we have the GetSetScaleParam descriptor class,
     but it does illustrate the functionality.)
    """

    def __init__(self, getter, setter=None, group="required", doc=None, update_on_set=True):
        self.getter = getter
        self.setter = setter
        self.__doc__ = doc
        if not group in ("required", "size", "optional"):
            raise TypeError("group keyword must be one of 'required', 'size' or 'optional'")
        else:
            self.group = group
        self.update_on_set = update_on_set
    
    def __get__(self, instance, cls):
        if instance is not None:
            return self.getter(instance)
        return self

    def __set__(self, instance, value):
        if not self.setter:
            raise TypeError("Cannot set parameter")
        self.setter(instance, value)
        if self.update_on_set:
            instance._SBProfile = None # Make sure that the ._SBProfile storage is emptied


class GetSetScaleParam(object):
    """
    Descriptor that uses a user-supplied scaling factor to get/set values, intended for defining
    "derived" quantities based on other parameters.

    Initialized with a name, root_name and factor argument, the descriptor behaves as follows...

    On get:
    >>> self.name
    ...returns self.root_name * factor

    On set:
    >>> self.name = value
    ...performs self.root_name = value / factor

    Use it like this in your classes:

    class MyProfile(GSObject):

        half_light_radius = SimpleParam("half_light_radius")
        fwhm = GetSetScaleParam("fwhm", "half_light_radius", RADIUS_CONVERSION_FACTOR)

    (N.B. The example above is functionally equivalent to the example given in the GetSetFuncParam
     docstring, and illustrates the neater syntax of this descriptor for this specific task.)
    
    """

    def __init__(self, name, root_name, factor, group="required", doc=None, update_on_set=True):
        self.name = name
        self.root_name = root_name
        self.factor = factor
        self.__doc__ = doc
        if not group in ("required", "size", "optional"):
            raise TypeError("group keyword must be one of 'required', 'size' or 'optional'")
        else:
            self.group = group
        self.update_on_set = update_on_set
    
    def __get__(self, instance, cls):
        if instance is not None:
            return instance._data[self.root_name] * self.factor
        return self

    def __set__(self, instance, value):
        instance._data[self.root_name] = value / self.factor
        if self.update_on_set:
            instance._SBProfile = None # Make sure that the ._SBProfile storage is emptied
        

class FluxParam(object):
    """
    A descriptor for storing and updating the flux parameter of a GSObject.

    Unlike SimpleParam this does not cause the GSObject's stored SBProfile to become undefined
    necessitating later re-initializtion, but rather calls the SBProfile's own setFlux() method
    to update the flux.

    This causes the SBProfile remain or become an SBTransform, and therefore not necessarily of the
    same object type as might be expected from the container GSObject.  However, all of the original
    GSObject params are available via their descriptors.
    """

    def __init__(self, default=1., group="optional", doc="Total flux of this object."):
        self.name = "flux"
        self.default = default
        self.__doc__ = doc
        if not group in ("required", "size", "optional"):
            raise TypeError("group keyword must be one of 'required', 'size' or 'optional'")
        else:
            self.group = group

    def __get__(self, instance, cls):
        if instance is not None:
            # dict.setdefault will return the item in the dict if present, or set and return the
            # default otherwise
            return instance._data.setdefault(self.name, self.default)
        return self

    def __set__(self, instance, value):
        # update the stored flux value
        instance._data["flux"] = value
        # update the SBProfile (do not undefine for re-initialization as do, e.g., SimpleParams).
        instance.SBProfile.setFlux(value)
        
