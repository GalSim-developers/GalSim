
Images and Related Concepts
###########################

The main purpose of GalSim is normally to create images that simulate real astronomical
observations.  As such, the `Image` class is the most common class in GalSim that you will
likely be working with.

An image if fundamentally an array of pixel values plus a bounding box indicating the range
of position values for those pixels, and a definition for how the image coordinates relate
to positions on the sky.

For the former concept, the `Bounds` class defines which x and y values are contained in 
the image, and the `Position` class is used to describe a particular location on an image.

For the latter concept, the simplest option is to just set a uniform pixel scale, e.g.
in arcsec/pixel.  But a wide range of more complicated `World Coordinate Systems` are
also possible.

.. toctree::
    :maxdepth: 2

    image_class
    bounds
    pos

