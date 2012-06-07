"""
Machinery (base classes, metaclasses, fields, etc.) used to
define a hierarchy of configuration options.
"""

class Field(object):
    """
    A class for custom Python "descriptors" (i.e. custom properties) used in configuration classes.

    A descriptor is an attribute of a class (NOT an attribute of an instance of a class);
    its __get__ and __set__ methods are called when the user tries to get or set
    an attribute of the same name on an instance of the class.  Because it is an attribute
    of the class, it can't store its data within itself, and instead stores it within
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
        if isinstance(instance, NodeBase):
            value = instance._data.setdefault(self.name, self.default)
            if value is True and self.type is not bool:
                # This combination is interpreted as "default construct on first access"
                value = self.type()
                instance._data[self.name] = value
            if issubclass(self.type, NodeBase) and not value.path:
                value.path = instance.path + (self.name,)
            return value
        return self

    def __set__(self, instance, value):
        if value is None:
            if self.required:
                raise TypeError("Cannot set required field '%s' to None." 
                                % self._get_full_name(instance))
        if issubclass(self.type, NodeBase):
            if issubclass(value, NodeBase):
                # setting 'outer.inner = Foo' is treated like 'outer.inner = Foo()'; the former
                # might be considered a nicer config syntax, even though it's weird if you
                # think of it like Python.
                value = value()
            if not isinstance(value, self.type):
                raise TypeError("Cannot set field '%s' of type '%s' to an instance of type '%s'."
                                % (self._get_full_name(instance), self.type.__name__, 
                                   type(value).__name__))
            self._update_node_path(instance, value, self.name)
            instance._data[self.name] = value
        else:
            try:
                instance._data[self.name] = self.type(value)
            except Exception as err:
                raise TypeError("Cannot set field '%s' of type '%s' to value '%s'"
                                % (self._get_full_name(instance), self.type.__name__, value))

    @staticmethod
    def _update_node_path(instance, value, name):
        if value.path:
            raise NotImplementedError("Relocating/copying nodes is not currently supported.")
        value.path = instance.path + (name,)

    def _get_full_name(self, instance):
        return ".".join(instance.path + (self.name,))

class NodeMeta(type):
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

        Note that 'self' in this method is the new Python class object we're creating.
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

class NodeBase(object):
    """
    Base class for all non-leaf configuration nodes.
    """

    # a dictionary of variables to make magically available when loading config files
    context = {}

    __metaclass__ = NodeMeta

    # derived classes should also define __slots__ to disable setting arbitrary attributes
    __slots__ = ("_data", "path")

    def __new__(cls, *args, **kwds):
        """
        First constructor for node initialization.

        Derived classes must always call the base class __new__ first, and then initialize
        any additional member variables.

        Keyword arguments corresponding to fields should not be processed in __new__.
        """
        self = object.__new__(cls, *args, **kwds)
        self._data = {}
        self.path = ()
        return self

    def __init__(self, *args, **kwds):
        """
        Second constructor for node initialization.

        This implementation attempts to set an attribute on self for each keyword argument.

        Derived classes should override any default values for fields BEFORE calling their
        base class __init__, and remove any keywords arguments that do not correspond
        to field values before passing them to the base class __init__.  Note that it
        isn't necessary to set defaults that match the default values set in the fields
        themselves; defaults set in __init__ override those values (which might be
        useful in a derived class of a node, or to set different defaults in a child
        node.
        """
        for k, v in kwds.iteritems():
            setattr(self, k, v)

    @property
    def name(self):
        """
        A '.'-separated string containing the full path of this config instance.
        """
        return ".".join(self.path)

    def load(self, filename):
        """
        Modify self by executing the given file, which should contain statements
        that modify a configuration root node named 'config' that has been mapped
        to self.
        """
        if issubclass(self, NodeBase):
            # we're calling this on a class not an instance, so we default-construct
            self = self()
        execfile(filename, globals={}, locals={"config":self})

    def reset(self):
        """
        Recursively reset all fields to their default values.
        """
        # we lazy-evaluate all fields anyhow, so we just need to clear the data dict
        # to make it appear like they've been set to their defaults
        self._data.clear()

    def finish(self, **kwds):
        """
        Prepare the config hierarchy to be used to process a catalog.

        Subclasses should set any nested fields that have computable defaults, setup
        any random number generators by passing them the given UniformDeviate, and
        raise an exception if any parameters are invalid.

        Keyword arguments will be passed to nested nodes; some keywords arguments may
        be necessary for some nodes, and subclasses should ignore and forward keywords
        they don't use.

        Base implementation iterates over all child nodes and calls finish() on them.
        """
        for name, field in self.fields.iteritems():
            value = getattr(self, name)   # this also ensures lazy-evaluated fields are evaluated
            if isinstance(value, NodeBase):
                value.finish(**kwds)

class ListNodeBase(NodeBase):
    """
    A base class for config nodes that are lists (including lists of nodes).
    """

    choices = None  # valid types for list elements, to be set by derived classes; ignored if None.

    __slots__ = ("_elements",)

    def __new__(cls, *args, **kwds):
        self = NodeBase.__new__(cls, *args, **kwds)
        self._elements = list()
        return self

    def _get_types(self):
        return tuple(type(e) for e in self._elements)

    def _set_types(self, types):
        values = []
        for index, cls in enumerate(types):
            element = cls()
            self._prep_insert(index, element)
            self._elements.append(element)
        self._elements = tuple(values)

    types = property(_get_types, _set_types, "sequence containing the types of elements")

    def _prep_insert(self, index, element):
        if self.choices is not None and not isinstance(element, self.choices):
            raise TypeError("Cannot set element %d of field '%s' to value '%s'"
                            % (index, self.name, element))
        if isinstance(element, NodeBase):
            if element.path:
                raise NotImplementedError("Relocating/copying nodes is not currently supported.")
            element.path = self.path[:-1] + ("%s[%d]" % (self.path[-1], index),)

    def __getitem__(self, index):
        return self._elements[index]

    def __setitem__(self, index, value):
        self._prep_insert(value)
        if index == len(self._elements): # implicit append by assigning to the next item
            self._elements.append(value)
        else:
            self._elements[index] = value

    def __iter__(self):
        return iter(self._elements)

    def __len__(self):
        return len(self._elements)

    def append(self, value):
        self._prep_insert(value)
        self._elements.append(value)

    def finish(self, **kwds):
        """
        Prepare the config hierarchy to be used to process a catalog.

        See NodeBase.finish() for additional documentation.

        This implementation simply calls finish on any list elements that are nodes.
        """
        NodeBase.finish(self, **kwds)
        for element in self:
            if isinstance(element, NodeBase):
                element.finish(**kwds)

def nested(*args, **kwds):
    """
    A decorator that  can be used to define a nested node class and add it as a field
    in one step.  It's used like this:

    class Outer(NodeBase):
        ...
        @nested(*args, **kwds)
        class inner(NodeBase):
            ...

    This is roughly equivalent to:

    class Outer(NodeBase):
        ...
        class inner(NodeBase):
            ...
        inner = Field(inner, *args, **kwds)

    One can also omit the parenthesis if no arguments are passed:

    class Outer(NodeBase):
        ...
        @nested
        class inner(NodeBase):
            ...

    In all cases, the docstring of the inner class will be used as the field
    documentation unless a 'doc' keyword argument is passed to the decorator.
    """
    if kwds or len(args) != 1 or not issubclass(args[0], NodeBase):
        kwds.setdefault("doc", cls.__doc__)
        def decorate(cls):
            return Field(cls, *args, **kwds)
        return decorate
    else:
        cls = args[0]
        return Field(cls, doc=cls.__doc__)
