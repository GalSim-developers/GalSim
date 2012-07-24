

class SimpleParam(object):
    """
    Descriptor that gets/sets a value, and on setting causes the GSObject's stored SBProfile to
    to be undefined for later re-initialization with the updated parameter set.

    Use it like this:

    class MyProfile(GSObject):
        flux = SimpleParam("flux")
    """

    def __init__(self, name, default=None, doc=None):
        self.name = name
        self.default = default
        self.__doc__ = doc

    def __get__(self, instance, cls):
        if instance is not None:
            # dict.setdefault will return the item in the dict if present, or set and return the default
            # otherwise
            return instance._data.setdefault(self.name, self.default)
        return self

    def __set__(self, instance, value):
        instance._data[self.name] = value
        instance._SBProfile = None

class GetSetParam(object):
    """
    Descriptor that uses user-supplied functions to get/set values, intended for
    defining "derived" quantities.

    Like SimpleParam, on setting this descriptor causes the GSObject's stored SBProfile to be
    undefined, for later re-initialization with the updated parameter set.

    Use it like this:

    class MyProfile(GSObject):
    
        half_light_radius = SimpleParam("half_light_radius")

        def _get_fwhm(self):
            return self.half_light_radius * RADIUS_CONVERSION_FACTOR
        def _set_fwhm(self, value):
            self.half_light_radius = value / RADIUS_CONVERSION_FACTOR
        fwhm = GetSetParam(_get_fwhm, _set_fwhm)
    """

    def __init__(self, getter, setter=None, doc=None):
        self.getter = getter
        self.setter = setter
        self.__doc__ = doc
    
    def __get__(self, instance, cls):
        if instance is not None:
            return self.getter(instance)
        return self

    def __set__(self, instance, value):
        if not self.setter:
            raise TypeError("Cannot set parameter")
        self.setter(instance, value)
        instance._SBProfile = None # Make sure that the ._SBProfile storage is emptied

class FluxParam(object)


    
