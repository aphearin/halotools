#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line script to download the alternate halo catalogs."""

import sys
import os
from halotools.sim_manager import CatalogManager

existing_fname_error_msg = ("\n\nThe following filename already exists in your cache directory: \n\n%s\n\n"
    "If you really want to overwrite the file, \n"
    "simply execute this script again but using ``-overwrite`` as a command-line argument.\n\n")

def main(command_line_args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, prog=os.path.basename(sys.argv[0]))
    allowed_simname = ('bolshoi','bolplanck','multidark','consuelo')
    allowed_redshift = (0,0.5,1,2)
    allowed_halofinder = ('bdm','rockstar')
    parser.add_argument('simname', metavar='SIMNAME', type=str,
                        choices=allowed_simname,
                        help='The name of the simulation to download, allowed values: {}.'.format(allowed_simname))
    parser.add_argument('redshift', metavar='REDSHIFT', type=float,
                        choices=allowed_redshift,
                        help='The redshift of the simulation snapshot to download, allowed values: {}.'.format(allowed_redshift))
    parser.add_argument('halo_finder', metavar='HALO_FINDER', type=str, nargs='?',
                        choices=allowed_halofinder, default=None,
                        help='The name of the halo-finder to download , allowed values: {} (ignored with --particlesonly).'.format(allowed_halofinder))

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--particlesonly',action='store_true',dest='particlesonly',
                        help="Only download the dark matter particles (overrides HALO_FINDER, conflicts with --halosonly).")
    group.add_argument('--halosonly',action='store_true',dest='halosonly',
                        help="Do not download the random downsamling of dark matter particles with the halo (conflicts with --particlesonly).")

    parser.add_argument('-o','--overwrite',action='store_true',dest='overwrite',
                        help="Overwrite the existing files in your cache directory.")
    # parser.add_argument('-v','--verbose',action='store_true',dest='verbose',
    #                     help='Print lots of extra output.')
    args = parser.parse_args()

    catman = CatalogManager()

    if args.halo_finder is None and not args.particlesonly:
        raise parser.error('You must specify either HALO_FINDER or --particlesonly.')
    if args.halo_finder and args.particlesonly:
        print("Ignoring halo_finder={}, and only downloading particles.".format(args.halo_finder))
        args.halo_finder = None

    msg = ("\nYour Halotools cache directory now has {number} hdf5 file(s),\n"
           "storing {fileinfo} for the {args.simname} simulation.\n"
           "The data structure stored by these hdf5 files is an Astropy Table.\n"
           "\nHalotools can load these catalogs into memory with the following syntax:\n\n"
           ">>> from halotools.sim_manager import HaloCatalog\n"
           ">>> halocat = HaloCatalog(simname='{args.simname}', redshift={args.redshift}{halo_arg})\n"
           "{python_example}\n")

    halo_str = "a z={args.redshift:0.2f} {args.halo_finder} halo catalog".format(args=args)
    particle_str = "a random downsampling of ~1e6 dark matter particles"
    halo_arg = ", halo_finder='{args.halo_finder}'".format(args=args)
    halos_example = ">>> halos = halocat.halo_table\n"
    particles_example = ">>> particles = halocat.ptcl_table\n"
    if args.halosonly:
        python_example = halos_example
        fileinfo = halo_str
        number = 1
    elif args.particlesonly:
        python_example = particles_example
        fileinfo = particle_str
        halo_arg = ""
        number = 1
    else:
        'for the {args.simname} simulation'
        python_example = halos_example + particles_example
        fileinfo = ' and '.join((halo_str,particle_str))
        number = 2

    msg = msg.format(args=args,python_example=python_example,fileinfo=fileinfo,halo_arg=halo_arg,number=number)

    if not args.particlesonly:
        catman.download_processed_halo_table(simname = args.simname,
            halo_finder = args.halo_finder, desired_redshift = args.redshift,
            initial_download_script_msg = existing_fname_error_msg,
            overwrite = args.overwrite, success_msg = msg)
    if not args.halosonly:
        catman.download_ptcl_table(simname = args.simname, 
            desired_redshift = args.redshift, dz_tol = 0.05, success_msg = msg, 
            initial_download_script_msg = existing_fname_error_msg, overwrite=args.overwrite)


###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
        main(sys.argv)

