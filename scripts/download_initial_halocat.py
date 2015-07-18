#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line script to download the default halo catalog"""

from halotools import sim_manager

simname = sim_manager.sim_defaults.default_simname
halo_finder = sim_manager.sim_defaults.default_halo_finder
redshift = sim_manager.sim_defaults.default_redshift

catman = sim_manager.read_nbody.CatalogManager()

catman.download_preprocessed_halo_catalog(simname, halo_finder, redshift)


