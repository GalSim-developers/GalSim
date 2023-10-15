# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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

# Based on https://github.com/sphinx-doc/sphinx/issues/823

import galsim

# The short X.Y version
version = '.'.join(map(str,galsim.__version_info__[:2]))

blob_url = 'https://github.com/GalSim-developers/GalSim/blob/releases/' + version + '/'

def gh_link_role(rolename, rawtext, text, lineno, inliner,
                 options={}, content=()):
    from docutils import nodes, utils
    name, path = text.split('<')
    path = path.split('>')[0]
    full_url = blob_url + path
    pnode = nodes.reference(internal=False, refuri=full_url)
    pnode += nodes.literal(name, name, classes=['file'])
    return [pnode], []


def setup(app):
    app.add_role('gh-link', gh_link_role)
