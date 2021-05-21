# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

from .main import run_main

if __name__ == '__main__':
    # The galsim executable will import this and then call run_main().
    # So in that case, we don't want to also run_main() here, since then it would run twice.
    # However, `python -m galsim config.yaml` will run this as a program, so then we do want
    # to call main.  Hence this `__name__ == '__main__'` block.
    run_main()
