DOXYGEN INTRO
=============

Basics
------

The GalSim code is documented using the Doxygen code comment system.  This is a structured form of
comment that can be parsed into attractive and useful forms by the doxygen program.  We will be 
commenting both the python and C++ codes with this system, and the syntax is only slightly 
different[0].

I have installed a helper program into docs/doxypy.py which makes it possible to use the doxygen 
comment special symbols in python docstrings, removing any need for duplication.

There are many different options for structuring doxygen comments - I suggest we stick with the one 
described here.

While Doxygen supports both "\" and "@" as a markup symbol, we recommend "@", because "\" is also 
used as an escape character in many other contexts.

The examples briefly shown here are fleshed out a little in 
docs/astronomer.py and docs/astronomy.hpp.  Run doxygen in the docs directory to generate  
docs/doxygen_example_output.  You can then open  docs/doxygen_example_output/index.html in a 
browser.

Running Doxygen
-------------
To build the GalSim documentation, install doxygen (available in all good package managers) and run 
the command "doxygen" in the main galsim directory.  The documentation will be built in the 
docs/html and docs/latex directories.



     C++
==============

For clarity, we will be putting our documentation comments in our C++ code in the header files, 
and not the code files.  This does not excuse you from commenting your code!  

Classes, methods, functions, files, and data declarations can all be doxy'd.  In all cases the 
comments appear directly above the thing they document, and start with a slash and two astrisks,
for example:


DOCUMENTING CLASSES AND STRUCTS
-------------------------------

/**
 *  @brief A telescope with a location and optical properties.
 *
 *  A telescope object simulates observations that are possible from its latitude and longitude.
 *  It has properties which specify the type of observations it can make and the colour the front
 *  door of the observatory is painted.  The main method to use on a Telescope is "observe".
 */
class Telescope
{
}

The gap line (with only '*', though we could equally have left it blank or used a different 
comment style) separates the description out into a brief overview description and a more detailed 
one.


DOCUMENTING FILES
-----------------
A description of the purpose or usage of the whole file can be given at the top of it, using the 
same format as the class documentation, but with the @file keyword and the current filename 
specified:

/**
 *  @file Astronomy.hpp
 *
 *  @brief Contains astronomy functions except those involving individual astronomers; this
 *         brief documentation spans more than one line.
 *
 *  All functions are in Astronomy namespace. Constants useful for observational cosmology
 *  are also supplied
 */




DOCUMENTING METHODS AND FUNCTIONS
---------------------------------

In GalSim all methods, including private ones, will be checked for documentation[1].  The 
parameters and return type of a method or function can be documented with the @param and @return 
lines.  The [in], [out], [in,out] modifiers to @param are useful to make it clear the purpose of 
function arguments where this is necessary. If the usage is obvious from the arguments themselves 
then they may be omitted rather than increase clutter.

/**
 *  @brief Construct a telescope at a location
 *
 *  Construct a telescope at the specified location parameters.  The optical properties and 
 *  interior decor will all use default values.
 *  @param[in] latitude    The latitude in degrees of the telescope
 *  @param[in] longitude   The longitude in degrees of the telescope (west is positive)
 *  @param[in] altitude    The telescope altitude above local sea level in meters.
 *  @return A telescope instance
 */
Telescope(float latitude, float longitude, float altitude);


DOCUMENTING CLASS/STRUCT ATTRIBUTES AND OTHER VARIABLES
------------------------------------------------
Variables belonging to a class or at global scope[2] can use the same documentation system as 
classes, etc, but can also be briefly documented on a single line with three slashes and a 
less-than sign:

float opacity;  ///< The mean nightly atmospheric opacity at the telescope intended wavelength 

This can also be used on #define quantities:
#define PI 3.13 ///< Value of pi, accurate enough for precision cosmology


     PYTHON
=================

CLASSES, METHODS, AND FUNCTIONS
-------------------------------

Since python has an in-build documentation string concept, doxygen can parse those strings to make 
its docs.  The docstrings of classes, functions and methods will be parsed straightforwardly.  
As in C++, the first line is a brief description and the rest are a more detailed one, or 
parameters or return types.  Since python is dynamically typed and does not enforce what type of 
object function arguments are it is a good idea to put in the docstring of functions and methods 
what type of objects you typically expect it to be called with.  As in C++, the @param and @return 
keywords indicate a description of parameters and return values.

class Astronomer(object):
    """A single Astronomer.
    
    An astronomer has any number of paper and exactly one name.
    It has a specialism, height, and age, and lifespan which is determined 
    primarily by its height.  Astronomers use Telescope object to observe.
    """
    def __init__(self, name, papers=None):
        """
        Construct an astronomer.
        
        Make up an astronomer object by specifying its components. All the normal 
        properties of astronomers are assumed.
        
        @param name (String) Mandatory - the astronomer's surname
        @param papers (List of strings) Optional - a list of papers by the astronomer
        @return (Astronomer) A new astronomer instance
        """


DOCUMENTING FILES
-----------------
The file docstring (which must be at the very top) is also parsed as the documentation for the 
whole file, but as with C++ you must use the @file keyword and specify the current filename to 
get this to work:
"""
@file astronomer.py Contains Astronomer class and constants

The constants are mainly default parameters for astronomer characteristics, like age and height.
"""

DOCUMENTING VARIABLES
---------------------
Since variables, including class and instance variables, do not have python docstrings, they have a 
different sytax in doxygen.  Unfortunately that does mean that if you want to describe your 
variables in the file docstring (so it can be read in python interactive sessions) then you need 
to type some things twice.  

The lines above the assignment of the variables should start with a double # comment:

##The assumed age of an astromomer in years
#
##This value is assumed for all astronomers unless an alternative value is chosen for them
DEFAULT_ASTRONOMER_AGE = 35

As always the first line is a brief description and subsequent ones (after the single #) are the 
detailed one.

INSTANCE VARIABLES
------------------
Since python instance variables (i.e. members of a class) do not need to be specified in advance,
they can be declared anywhere in the methods of the class.  Like with file-level variables this may 
mean you repeat yourself in the docstring).  

No matter where they are the in these functions you can write the same format comment as with other 
variables directly above where they are first assigned.  It is most sensible to do this in the init 
(constructor) method.  


def __init__(self, name, ...):
...
...
...
    ##(String) The astronomer surname
    #
    ## The astronomer surname including any hyphenation.
    self.name=name  




BEST PRACTICE HINTS
===================

Because this is a developer-focued project we should document both private and public methods.  In 
python, note in the description if a method is mainly internal and not used by other classes.  Do 
the same in C++ if a class should probably not be used directly from outside a particular part of 
the code.

You should describe the typical use for a function or class (e.g. "Usually instantiated by an 
SBProfile object", "should be passed to the optics module to generate output, etc.")

In python, briefly describe what types the input and output to a class are expected to be by 
starting the description with [string],  [integer], or [float], or whatever.

You don't need to described *how* a function or method achieves what it does unless it has 
significant interaction with other functions or classes - you can leave that to the code comments.
Use the doxygen comment to describe *what* it does.

If an input to a function is modified by the function and this is not immediately obvious it is a 
good idea to note this in the docs.

If you only have limited time, focus on the functions that will be end users of the code.

If a function input may have units, say what they are expected to be  (e.g. degrees or radians 
for an angle).



[0] Oddly enough choosing the OPTIMIZE_FOR_JAVA option in the Doxyfile seems to be the best way to 
nicely document both C++ and Python at once.
[1] This is an option in the Doxyfile that I have set.
[2] But please don't put many variables at global scope.  That's usually a bad sign.

