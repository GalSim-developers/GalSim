Changes from v0.3 to current version: 
------------------------------------

* When making GSObjects out of real images that have noise, it is possible to pad those images with
  a noise field (either correlated or uncorrelated) so that there is not an abrupt change of
  properties in the noise field when crossing the border into the padding region.  (Issue #238)
