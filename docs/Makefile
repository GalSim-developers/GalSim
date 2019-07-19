# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
# The DYLD... below is for users who use the LSST stack for GalSim dependencies.
# We think it doesn't break anything for non-LSST-stack users, but if you find
# you have problems building the documentation, try replacing this line with just
# SPHINXBUILD = sphinx-build
SPHINXBUILD   = DYLD_LIBRARY_PATH=$$LSST_LIBRARY_PATH sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
