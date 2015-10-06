#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line script to download the default halo catalog."""

import sys
import os
from halotools.sim_manager import CatalogManager, sim_defaults

existing_fname_error_msg = ("\n\nThe following filename already exists in your cache directory: \n\n%s\n\n"
    "If you really want to overwrite the file, \n"
    "simply execute this script again but using ``-overwrite`` as a command-line argument.\n\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, prog=os.path.basename(sys.argv[0]))
    # parser.add_argument('module', metavar='MODULE', type=str,
    #                     help='The name of the module to download.')
    parser.add_argument('-o','--overwrite',action='store_true',dest='overwrite',
                        help="Overwrite the existing file with a new download.")
    # parser.add_argument('-v','--verbose',action='store_true',dest='verbose',
    #                     help='Print lots of extra output.')
    args = parser.parse_args()

    simname = sim_defaults.default_simname
    halo_finder = sim_defaults.default_halo_finder
    redshift = sim_defaults.default_redshift

    catman = CatalogManager()

    catman.download_processed_halo_table(simname = simname, 
        halo_finder = halo_finder, desired_redshift = redshift, 
        initial_download_script_msg = existing_fname_error_msg, overwrite=args.overwrite)
    catman.download_ptcl_table(simname = simname, 
        desired_redshift = redshift, dz_tol = 0.05, overwrite=args.overwrite)

    msg = ("\n\nYour Halotools cache directory now has two hdf5 files, \n"
        "one storing a z = %.2f %s halo catalog for the %s simulation, \n"
        "another storing a random downsampling of ~1e6 dark matter particles from the same snapshot.\n"
        "\nHalotools can load these catalogs into memory with the following syntax:\n\n"
        ">>> from halotools.sim_manager import HaloCatalog\n"
        ">>> bolshoi_z0 = HaloCatalog()\n"
        ">>> halos = bolshoi_z0.halo_table\n"
        ">>> particles = bolshoi_z0.ptcl_table\n\n")

    print(msg % (abs(redshift), halo_finder, simname))


###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
        main()

