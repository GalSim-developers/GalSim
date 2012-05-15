"""
Machinery (base classes, metaclasses, fields, etc.) used to
define a hierarchy of configuration options.
"""

class FieldBase(object):
    """
    A base class for custom Python "descriptors" (i.e. custom properties) used in
    configuration classes.

    A descriptor is an attribute of a class (NOT an attribute of an instance of a class);
    its __get__ and __set__ methods are called when the user tries to get or set
    an attribute of the same name on an instance of the class.  Because it is an attribute
    of the class, it can't store its data within itself, and instance stores it within
    a dictionary called "_data" in the instance.
    """

    def __init__(self, type, default=True, required=False, doc=None):
        self.type = type
        self.default = default
        self.required = required
        if doc is None:
            doc = self.type.__doc__
        self.__doc__ = doc

    def __get__(self, instance, type):
        if isinstance(instance, ConfigBase):
            value = instance._data.setdefault(self.name, self.default)
            if value is True and self.type is not bool:
                # This combination is interpreted as "default construct on first access"
                value = self.type()
                instance._data[self.name] = value
            if issubclass(self.type, ConfigBase) and not value.path:
                value.path = instance.path + (self.name,)
            return value
        return self

class ConfigMeta(type):
    """
    A metaclass for configuration classes.  A metaclass is a custom type for a class.
    """

    def __init__(self, name, bases, dict):
        """
        The __init__ function is called when a new class with this metaclass is defined,
        allowing us to execute code at that point.  Here, we add some special handling
        for any FieldBase objects we find among the class attributes, creating a dict
        of all fields so we can iterate over them later and setting the field's name
        attribute.
        """
        type.__init__(self, name, bases, dict)
        self.fields = {}
        for k, v in self.__dict__.iteritems():
            if isinstance(v, FieldBase):
                self.fields[k] = v
                v.name = k

    def __setattr__(self, name, value):
        """
        Ensure we also handle fields added to a class after its main definition block.
        """
        if isinstance(value, FieldBase):
            self.fields[name] = value
            value.name = name
        type.__setattr__(self, name, value)        

    def __call__(self, *args, **kwds):
        """
        This is called when a new instance of the class is constructed;
        it's responsible for actually calling the constructors.  In between
        the call to __new__ and __init__, we initialize the ._data dictionary
        with all the defaults from the fields.
        """
        instance = self.__new__(self, *args, **kwds)
        instance._data = {}
        instance.path = ()
        instance.__init__(*args, **kwds)
        return instance

class ConfigBase(object):
    """
    Base class for all non-leaf configuration nodes.
    """

    __metaclass__ = ConfigMeta

    @property
    def name(self):
        """A '.'-separated string containing the full path of this config instance."""
        return ".".join(self.path)

class Field(FieldBase):

    def __set__(self, instance, value):
        if value is None:
            if self.required:
                raise TypeError("Cannot set required field '%s' to None." 
                                % self._get_full_name(instance))
        if issubclass(self.type, ConfigBase):
            if not isinstance(value, self.type):
                raise TypeError("Cannot set field '%s' of type '%s' to an instance of type '%s'."
                                % (self._get_full_name(instance), self.type.__name__, 
                                   type(value).__name__))
            if value.path:
                raise NotImplementedError("Relocating/copying nodes is not currently supported.")
            value.path = instance.path + (self.name,)
            instance._data[self.name] = value
        else:
            try:
                instance._data[self.name] = self.type(value)
            except Exception as err:
                raise TypeError("Cannot set field '%s' of type '%s' to value '%s'"
                                % (self._get_full_name(instance), self.type.__name__, value))

    def _get_full_name(self, instance):
        return ".".join(instance.path + (self.name,))
