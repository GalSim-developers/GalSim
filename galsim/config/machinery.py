"""
Machinery (base classes, metaclasses, fields, etc.) used to
define a hierarchy of configuration options.
"""

def _set_node_impl(container, key, types, aliases, value, path):
    value = aliases.get(value, value)
    if value is None:
        container[key] = value
        return value
    isNode = False
    if issubclass(NodeBase, value):
        try:
            value = value()
        except:
            raise TypeError("could not default-construct instance")
        isNode = True
    elif isinstance(NodeBase, value):
        isNode = True
    if types is not None and not isinstance(value, types):
        raise TypeError("invalid type for this field")
    if isNode:
        if value.path:
            raise TypeError("moving/copying nodes is not currently supported")
        value.path = path
    container[key] = value
    return value

class Field(object):
    """
    A class for custom Python "descriptors" (i.e. custom properties) used in configuration classes.

    A descriptor is an attribute of a class (NOT an attribute of an instance of a class);
    its __get__ and __set__ methods are called when the user tries to get or set
    an attribute of the same name on an instance of the class.  Because it is an attribute
    of the class, it can't store its data within itself, and instead stores it within
    a dictionary called "_data" in the instance.
    """

    def __init__(self, types=(), default=True, doc=None, aliases=None, type=None):
        if types is None:
            types = ()
        types = tuple(types)
        if type is not None:
            types += (type,)
        self.types = types
        if aliases is None:
            aliases = {}
        self.aliases = aliases
        self.default = self.aliases.get(default, default)
        if doc is None:
            try:
                doc = self.types[0].__doc__
            except:
                pass
        self.__doc__ = doc

    def __get__(self, instance, cls):
        if isinstance(instance, NodeBase):
            try:
                value = instance._data[self.name]
            except KeyError:
                try:
                    value = _set_node_impl(instance._data, self.name, self.types, self.aliases, value,
                                           path=instance.path + (self.name,))
                except Exception, err:
                    raise TypeError("Error constructing default value for field '%s': %s"
                                    % (self._get_full_name(instance), err))
            return value
        return self

    def __set__(self, instance, value):
        try:
            _set_node_impl(instance._data, self.name, self.types, self.aliases, value,
                           path=instance.path + (self.name,))
        except Exception, err:
            raise TypeError("Error setting value of field '%s': %s"
                            % (self._get_full_name(instance), err))

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
        d = self._get_load_context()
        d["config"] = self
        d.update((cls.__name__, cls) for cls in generators.load_context)
        execfile(filename, globals={}, locals=d)

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

    @classmethod
    def _get_load_context(cls, output=None):
        """
        Construct a dictionary of all nested aliases.  This allows names to be used in config
        files without imports.
        """
        if output is None:
            output = {}
        for name, field in cls.fields.iteritems():
            for k, v in field.aliases.iteritems():
                if isinstance(k, basestring):
                    if output.setdefault(k, v) is not v:
                        raise RuntimeError("Alias conflict in config load context: %r" % k)
            for cls in field.types:
                try:
                    cls._get_load_context(output)
                except AttributeError:
                    pass
        return output

class ListNodeBase(NodeBase):
    """
    A base class for config nodes that are lists (including lists of nodes).
    """

    types = None  # valid types for list elements, to be set by derived classes; ignored if None.
    aliases = {}  # dictionary used to lookup aliases for field values

    __slots__ = ("_elements",)

    def __new__(cls, *args, **kwds):
        self = NodeBase.__new__(cls, *args, **kwds)
        self._elements = list()
        return self

    def __getitem__(self, index):
        return self._elements[index]

    def __setitem__(self, index, value):
        if type(index) is slice:
            if index.start is not None or index.stop is not None or index.step is not None:
                raise TypeError("Advanced slice set operations are not supported on list nodes")
            # If we didn't throw, we know we're just replacing the entire list; that's the only
            # slice set syntax we support.
            newList = [None] * len(value)    # operate on temporary for better exception recovery
            for n, element in enumerate(value):
                try:
                    _set_node_impl(newList, n, self.types, self.aliases, element,
                                   path=self.path[:-1] + ("%s[%d]" % (self.path[-1], index),))
                except Exception, err:
                    raise TypeError("Error setting element %d of field '%s': %s"
                                    % (n, self.name, err))
            self._elements = newList
        else:
            if index == len(self._elements): # implicit append by assigning to the next item
                self._elements.append(None)
            try:
                _set_node_impl(self._elements, index, self.types, self.aliases, element,
                               path=self.path[:-1] + ("%s[%d]" % (self.path[-1], index),))
            except Exception, err:
                raise TypeError("Error setting element %d of field '%s': %s"
                                % (index, self.name, err))

    def __iter__(self):
        return iter(self._elements)

    def __len__(self):
        return len(self._elements)

    def append(self, value):
        self._elements.append(None)
        try:
            _set_node_impl(self._elements, -1, self.types, self.aliases, element,
                           path=self.path[:-1] + ("%s[%d]" % (self.path[-1], index),))
        except Exception, err:
            raise TypeError("Error setting element %d of field '%s': %s"
                            % (index, self.name, err))

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

    @classmethod
    def _get_load_context(cls, output=None):
        """
        Construct a dictionary of all nested aliases.  This allows names to be used in config
        files without imports.
        """
        if output is None:
            output = {}
        NodeBase._get_load_context(output)
        if self.types is not None:
            for cls in self.types:
                try:
                    cls._get_load_context(output)
                except AttributeError:
                    pass
        return output

class ListField(Field):

    def __init__(self, types=(), default=True, doc=None, aliases=None, type=None, node_cls=None):
        pass #TODO

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
