# -*- coding: utf-8 -*-
"""
Methods and classes to load halo and particle catalogs into memory.

"""

__all__=['ProcessedSnapshot','CatalogManager', 'RockstarReader']

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import numpy as np
import os, sys, warnings, urllib2, fnmatch
import pickle
from time import time
import datetime

HAS_SOUP = False
try:
    from bs4 import BeautifulSoup
    HAS_SOUP = True
except:
    pass

HAS_REQUESTS = False
try:
    import requests
    HAS_REQUESTS = True
except:
    pass

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from astropy.table import Table

from . import cache_config, supported_sims, sim_defaults

from ..utils.array_utils import find_idx_nearest_val
from ..utils.array_utils import array_like_length as custom_len
from ..utils.io_utils import download_file_from_url


def get_halocat_obj(simname, halo_finder):
    """ Find and return the class instance that will be used to 
    convert raw ASCII halo catalog data into a reduced binary.

    Parameters 
    ----------
    simname : string 
        Nickname of the simulation, e.g., `bolshoi`. 

    halo_finder : string 
        Nickname of the halo-finder that generated the catalog, 
        e.g., `rockstar`. 

    Returns 
    -------
    halocat_obj : object 
        Class instance of `~halotools.sim_manager.supported_sims.HaloCat`. 
        Used to read ascii data in the specific format of the 
        `simname` simulation and `halo_finder` halos. 
    """
    class_list = supported_sims.__all__
    parent_class = supported_sims.HaloCat

    supported_halocat_classes = []
    for clname in class_list:
        clobj = getattr(supported_sims, clname)
        if (issubclass(clobj, parent_class)) & (clobj.__name__ != parent_class.__name__):
            supported_halocat_classes.append(clobj())

    halocat_obj = None
    for halocat in supported_halocat_classes:
        if (halocat.simname == simname) & (halocat.halo_finder == halo_finder):
            halocat_obj = halocat
    if halocat_obj is None:
        print("No simulation class found for %s simulation and %s halo-finder.\n"
            "Either there was a typo in specifying the simname and/or halo-finder,\n"
            "or you tried to use an unsupported halo catalog. \n"
            "If you want to use Halotools to manage and process halo catalogs, \n"
            "then you must either use an existing reader class "
            "defined in the supported_sims module, \nor you must write your own reader class.\n" 
            % (simname, halo_finder))
        return None
    else:
        return halocat_obj


class ProcessedSnapshot(object):
    """ Class containing halo and particle data taken from 
    a single snapshot of some Nbody simulation.
    """

    def __init__(self, simname=sim_defaults.default_simname, 
        halo_finder=sim_defaults.default_halo_finder,
        redshift = sim_defaults.default_redshift, verbose=False, 
        **kwargs):
        """
        Parameters 
        ----------
        simname : string, optional 
            Nickname of the simulation, e.g. `bolshoi`. 
            If no simname is specified, the default choice is set 
            by the ``default_simname`` string stored in 
            the `sim_defaults` module.

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. `rockstar`. 
            If no halo_finder is specified, the default choice is set 
            by the ``default_halo_finder`` string stored in 
            the `sim_defaults` module.

        redshift : float, optional 
            Redshift of the desired snapshot. 
            If no redshift is specified, the default choice is set 
            by the ``default_redshift`` string stored in 
            the `sim_defaults` module.

        version_name : string, optional
            For cases where multiple versions of the same halo catalog 
            are stored in the cache, 
            a matching version name must be supplied to disambiguate. 

        verbose : bool, optional 
            If True, a range of print statements will be issued as sanity checks 
            on bookkeeping. Default is False. 

        """

        self.simname = simname
        self.halo_finder = halo_finder

        self.catman = CatalogManager()
        result = self.catman.closest_halocat(
            'cache', 'halos', self.simname, self.halo_finder, redshift, 
            **kwargs
            )

        if result is None:
            raise IOError("No processed halo catalogs found in cache "
                " for simname = %s and halo-finder = %s" % (simname, halo_finder))
        else:
            self.halocat_fname, self.redshift = result[0], result[1]

        self.halocat_obj = get_halocat_obj(simname, halo_finder)
        self.Lbox = self.halocat_obj.simulation.Lbox
        self.particle_mass = self.halocat_obj.simulation.particle_mass
        self.softening_length = self.halocat_obj.simulation.softening_length
        self.cosmology = self.halocat_obj.simulation.cosmology

        self.particles = None
        self.halos = self.catman.load_halo_catalog(
            fname=self.halocat_fname, 
            simname = self.simname, 
            halo_finder = self.halo_finder, 
            redshift = self.redshift, 
            verbose=verbose, 
            **kwargs)

        self._bind_halocat_metadata()

    def _bind_halocat_metadata(self):
        f = h5py.File(self.halocat_fname)
        for key in f.attrs.keys():
            if type(f.attrs[key])==str:
                setattr(self, key, f.attrs[key])
            elif type(f.attrs[key])==dict:
                for dict_key in f.attrs[key]:
                    setattr(self, dict_key, f.attrs[key][dict_key])
        f.close()



###################################################################################################

class CatalogManager(object):
    """ Container class for managing I/O of halo & particle catalogs.
    """

    def __init__(self):
        pass

    @property 
    def available_halocats(self):
        """ Return a list of the names of all
         simulations and halo catalogs supported by Halotools. 

        Returns 
        -------
        supported_halocats : list
            List of 2-element tuples. Both tuple entries are strings. 
            The first tuple element gives the nickname of a simulation 
            supported by Halotools, e.g., `bolshoi`; 
            the second tuple element gives the name of a halo-finder 
            supported for that simulation, e.g., `rockstar`. 

        """
        class_list = supported_sims.__all__
        parent_class = supported_sims.HaloCat

        supported_halocats = []
        for clname in class_list:
            clobj = getattr(supported_sims, clname)
            if (issubclass(clobj, parent_class)) & (clobj.__name__ != parent_class.__name__):
                clinst = clobj()
                supported_halocats.append((clinst.simname, clinst.halo_finder) )

        return supported_halocats


    def available_snapshots(self, location, catalog_type, simname, halo_finder):
        """
        Return a list of the filenames of all snapshots that are stored at the input location. 

        Parameters 
        ----------
        location : string 
            Specifies the web or disk location to search for halo catalogs. 
            Optional values for `location` are:

                *  `web` - default web location defined by `~halotools.sim_manager.HaloCat` instance. 

                * `cache` - Halotools cache location defined in `~halotools.sim_manager.cache_config`

                * a full pathname such as `/explicit/full/path/to/my/personal/halocats/`. 

        catalog_type : string 
            If you want the original, unprocessed ASCII data produced by Rockstar, 
            then `catalog_type` should be set to `raw_halos`. 
            If you instead want a previously processed catalog that has been 
            converted into a fast-loading binary, set `catalog_type` to `halos`. 

        simname : string 
            Nickname of the simulation, e.g. `bolshoi`. 
            Default is None, in which case halo catalogs pertaining to all 
            simulations stored in `location` will be returned. 

        halo_finder : string
            Nickname of the halo-finder, e.g. `rockstar`. 
            Default is None, in which case halo catalogs pertaining to all 
            halo-finders stored in `location` will be returned. 

        Returns 
        -------
        fname_list : list 
            List of strings storing the filenames 
            (including absolute path) of all halo catalogs stored 
            at the input location. 

        """

        halocat_obj = get_halocat_obj(simname, halo_finder)
        if halocat_obj is None:
            return None

        if location == 'web':
            if catalog_type == 'raw_halos':
                return halocat_obj.raw_halocats_available_for_download
            elif catalog_type == 'halos':
                return halocat_obj.preprocessed_halocats_available_for_download
        else:
            if location == 'cache':
                dirname = cache_config.get_catalogs_dir(
                    catalog_type, simname=simname, halo_finder=halo_finder)
            else:
                dirname = location
                if not os.path.exists(dirname):
                    raise IOError("The following location passed as `location`"
                        " is not a directory:\n %s" % dirname)

            fname_list = []
            for (dirpath, dirnames, filenames) in os.walk(dirname):
                matching_fnames = fnmatch.filter(filenames, halocat_obj.halocat_fname_pattern)
                for f in matching_fnames:
                    fname_list.append(os.path.join(dirname,f))
            return fname_list

    def available_redshifts(self, location, catalog_type, simname, halo_finder):
        """
        Return a list of the redshifts that are stored at the input location. 

        Parameters 
        ----------
        location : string 
            Specifies the web or disk location to search for halo catalogs. 
            Optional values for `location` are:

                *  `web` - default web location defined by `~halotools.sim_manager.HaloCat` instance. 

                * `cache` - Halotools cache location defined in `~halotools.sim_manager.cache_config`

                * a full pathname such as `/explicit/full/path/to/my/personal/halocats/`. 

        catalog_type : string 
            If you want the original, unprocessed ASCII data produced by Rockstar, 
            then `catalog_type` should be set to `raw_halos`. 
            If you instead want a previously processed catalog that has been 
            converted into a fast-loading binary, set `catalog_type` to `halos`. 

        simname : string 
            Nickname of the simulation, e.g. `bolshoi`. 
            Default is None, in which case halo catalogs pertaining to all 
            simulations stored in `location` will be returned. 

        halo_finder : string
            Nickname of the halo-finder, e.g. `rockstar`. 
            Default is None, in which case halo catalogs pertaining to all 
            halo-finders stored in `location` will be returned. 

        Returns 
        -------
        redshift_list : list 
            List of redshifts of all halo catalogs stored 
            at the input location. 

        """

        halocat_obj = get_halocat_obj(simname, halo_finder)
        if halocat_obj is None:
            return None

        snapshot_list = self.available_snapshots(location, catalog_type, simname, halo_finder)
        redshift_list = []
        for full_fname in snapshot_list:
            fname = os.path.basename(full_fname)
            scale_factor_substr = halocat_obj.get_scale_factor_substring(fname)
            a = float(scale_factor_substr)
            z = (1/a) - 1
            redshift_list.append(z)

        return redshift_list


    def download_raw_halocat(self, simname, halo_finder, input_redshift, 
        overwrite = False, **kwargs):
        """ Method to download publicly available ascii data of 
        a specific raw halo catalog from its web location. 

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation, e.g. `bolshoi`. 

        halo_finder : string 
            Nickname of the halo-finder, e.g. `rockstar`. 

        input_redshift : float 
            Redshift of the requested snapshot. Must match one of the 
            available snapshots, or a prompt will be issued providing the nearest 
            available snapshots to choose from. 

        dz_tol : float, optional
            Tolerance value determining how close the requested redshift must be to 
            some available snapshot before issuing a warning. Default value is 0.1. 

        download_loc : string, optional
            Absolute pathname of where the raw halo catalog will be stored. 
            Default is the halotools cache directory. 

        overwrite : boolean, optional
            If a file with the same filename already exists 
            in the requested download location, the `overwrite` boolean determines 
            whether or not to overwrite the file. Default is False, in which case 
            no download will occur if a pre-existing file is detected. 

        Returns 
        -------
        output_fname : string  
            Filename (including absolute path) of the location of the downloaded 
            halo catalog.  
        """

        if HAS_SOUP == False:
            print("Must have BeautifulSoup installed to use Halotools Catalog Manager")
            return 

        if 'dz_tol' in kwargs.keys():
            dz_tol = kwargs['dz_tol']
        else:
            dz_tol = 0.1

        halocat_obj = get_halocat_obj(simname, halo_finder)
        list_of_available_snapshots = halocat_obj.raw_halocats_available_for_download
        closest_snapshot_fname, redshift_of_closest_match = (
            halocat_obj.closest_halocat(
            list_of_available_snapshots, input_redshift)
            )

        if abs(redshift_of_closest_match - input_redshift) > dz_tol:
            msg = (
                "No raw %s halo catalog has \na redshift within %.2f " + 
                "of the input_redshift = %.2f.\n The closest redshift for these catalogs is %.2f"
                )
            print(msg % (simname, dz_tol, input_redshift, redshift_of_closest_match))
            return 

        url = halocat_obj.raw_halocat_web_location + closest_snapshot_fname

        if 'download_loc' in kwargs.keys():
            download_loc = kwargs['download_loc']
            # We were given an explicit path to store the catalog
            # Check that this path actually exists, and if so, use it 
            if not os.path.exists(download_loc):
                raise IOError("Input directory name %s for download location"
                    "of raw halo catalog does not exist" % download_loc)
            else:
                output_fname = os.path.join(download_loc, closest_snapshot_fname)
        else:
            # We were not given an explicit path, so use the default Halotools cache dir
            cache_dirname = cache_config.get_catalogs_dir('raw_halos', 
                simname=simname, halo_finder=halo_finder)
            download_loc = cache_dirname
            output_fname = os.path.join(download_loc, closest_snapshot_fname)

        # Check whether there are existing catalogs matching the file pattern 
        # that is about to be downloaded
        is_in_cache = self.check_for_existing_halocat(
            download_loc, 'raw_halos', simname, halo_finder, 
            fname=output_fname)

        if is_in_cache != False:
            if overwrite ==True:
                warnings.warn("Downloading halo catalog and overwriting existing file %s" % output_fname)
            else:
                msg = ("The following filename already exists in your cache directory: \n\n%s\n\n"
                    "If you really want to overwrite the file, \n"
                    "you must call the same function again \n"
                    "with the keyword argument `overwrite` set to `True`")
                print(msg % output_fname)
                return None

        download_file_from_url(url, output_fname)

        return output_fname

    def process_raw_halocat(self, input_fname, simname, halo_finder, **kwargs):
        """ Method reads in raw halo catalog ASCII data, makes the desired cuts, 
        returns a numpy structured array of the rows passing the cuts, and optionally 
        stores the result as an hdf5 file in the cache directory. 

        Parameters 
        ----------
        input_fname : string 
            filename (including absolute path) where the ASCII data are stored. 

        simname : string 
            nickname of the simulation, e.g., `bolshoipl`. 

        halo_finder : string 
            Nickname of the halo-finder, e.g., `rockstar`. 

        cuts_funcobj : function object, optional
            Function used to apply cuts to the rows of the ASCII data. 
            `cuts_funcobj` should accept a structured array as input, 
            and return a boolean array of the same length. 
            If None, default cut is set by 
            `~halotools.sim_manager.RockstarReader.default_halocat_cut`. 

        store_result : bool, optional
            Determines whether the resulting structured array is stored to 
            an hdf5 file on disk. Default is False. If True, you must 
            at least pass the `version_name` keyword argument. 
            See the `store_processed_halocat` method for the other optional 
            keywords you may use when setting `store_result` to True. 

        notes : dict, optional 
            If you are storing the result as an hdf5 file, 
            the `notes` keyword argument give you the option to give 
            a qualitative description of how the catalog was produce. 
            `notes` should be a python dictionary. Each dict key of `notes` 
            will be stored hdf5 file as metadata, and will be the name of an 
            attribute of the corresponding `ProcessedSnapshot` object. 
            The value attached to each key can be any string. 

        Returns 
        -------
        arr : array 
            Structured numpy array storing all rows passing the cuts. 
            Column names of `arr` are determined by the 
            `~halotools.sim_manager.HaloCat.halocat_column_info` method of the 
            appropriate class instance of 
            `~halotools.sim_manager.HaloCat`. 

        reader_obj : object 
            Class instance of the reader used to process the raw halo data, e.g., 
            `~halotools.sim_manager.RockstarReader`. 
            If you want to use the `store_processed_halocat` method of this class 
            to manage the bookkeeping for the processed catalog, 
            it will be necessary to pass `reader_obj` as an input argument 
            to `store_processed_halocat`; 
            this requirement is used to ensure consistent bookkeeping.  

        """
        reader = RockstarReader(input_fname, 
            simname=simname, halo_finder=halo_finder, **kwargs)

#        if 'cuts_funcobj' not in kwargs.keys():
#            self.cuts_funcobj = kwargs['cuts_funcobj']
#        else:
#            self.cuts_funcobj = reader.default_halocat_cut
#            kwargs['cuts_funcobj'] = self.cuts_funcobj

        arr = reader.read_halocat(**kwargs)
        reader._compress_ascii()

        ### Calculation complete
        ### Now store the result, if applicable
        if 'store_result' in kwargs.keys():
            store_result = kwargs['store_result']
        else:
            store_result = False

        if store_result is True:
            if 'version_name' not in kwargs.keys():
                raise KeyError("If keyword argument store_result is True, "
                    "must also pass version_name keyword argument")
            else:
                version_name = kwargs['version_name']
                del kwargs['version_name']
                self.store_processed_halocat(
                    arr, reader, version_name, **kwargs)
            return arr, reader
        else:
            return arr, reader

    def store_processed_halocat(self, catalog, reader_obj, version_name, **kwargs):
        """
        Method stores an hdf5 binary of the reduced halo catalog to the desired location. 
        The resulting hdf5 file includes metadata 
        describing the source of the original halo catalog, the exact cuts that 
        were used to reduce the original catalog, the timestamp the reduced catalog was 
        originally processed, plus any optionally provided notes. 

        Parameters 
        ----------
        catalog : structured array 
            Numpy array of halo data. 
            Returned as the first output of `process_raw_halocat`. 

        reader_obj : object 
            Class instance of the reader used to reduce the raw ASCII data into 
            a structured numpy array, e.g., `~halotools.sim_manager.RockstarReader`. 
            Returned as the second output of `process_raw_halocat`. 

        version_name : string 
            String that will be appended to the original halo catalog filename
            to create a new filename for the cut halo catalog. 

        output_loc : string, optional
            Location to store catalog on disk. Default is Halotools cache directory. 
            (File sizes of the Halotools-supported 
            processed binaries typically vary from 100Mb-1Gb, depending on the cuts). 

        overwrite : bool, optional 
            If True, and if there exists a catalog with the same filename in the 
            output location, the existing catalog will be overwritten. Default is False. 

        cuts_funcobj : function object, optional
            Function used to apply cuts to the rows of the ASCII data. 
            `cuts_funcobj` should accept a structured array as input, 
            and return a boolean array of the same length. 
            If None, default cut is set by 
            `~halotools.sim_manager.RockstarReader.default_halocat_cut`. 
            The `store_processed_halocat` method will raise an exception if this 
            `cuts_funcobj` differs from the `cuts_funcobj` originally passed 
            to `reader_obj`. 

        notes : dict, optional 
            Additional notes that will be appended to the stored hdf5 file as metadata. 
            Each dict key of `notes` will be a metadata attribute of the hdf5 file, accessible 
            via hdf5_fileobj.attrs[key]. The value attached to each key can be any string. 

        Returns 
        -------
        output_full_fname : string 
            Filename (including absolute path) to the output hdf5 file. 
        """

        if HAS_H5PY==False:
            raise ImportError("Must have h5py installed to use the "
                "store_processed_halocat method")
            return 

        ##############################
        ### Interpret optional inputs 

        ### output location ###
        if 'output_loc' in kwargs.keys():
            output_loc = kwargs['output_loc']
        else:
            output_loc = 'cache'

        if output_loc == 'cache':
            output_loc = cache_config.get_catalogs_dir('halos', 
                simname=reader_obj.simname, halo_finder=reader_obj.halo_finder)
        else:
            if not os.path.exists(output_loc):
                raise IOError("The store_processed_halocat method "
                    "was passed the following output_loc argument: \n%s\n"
                    "This path does not exist. ")

        ### overwrite preference ###
        if 'overwrite' in kwargs.keys():
            overwrite = kwargs['overwrite']
        else:
            overwrite = False

        ### notes to attach to output hdf5 as metadata ###
        if 'notes' in kwargs.keys():
            notes = kwargs['notes']
        else:
            notes = {}
        for key, value in notes.iteritems():
            if type(value) != str:
                raise ValueError("Strings are the only permissible data types of values "
                    "attached to keys in the input notes dictionary")
        ##############################

        orig_catalog_fname = os.path.basename(reader_obj.fname)
        if orig_catalog_fname[-3:] == '.gz':
            orig_catalog_fname = orig_catalog_fname[:-3]

        output_fname = orig_catalog_fname + '.' + version_name + '.hdf5'
        output_full_fname = os.path.join(output_loc, output_fname)
        t = Table(catalog)
        print("Storing reduced halo catalog in the following location:\n" + 
            output_full_fname)
        t.write(output_full_fname, path='halos', overwrite=overwrite)

        #################################

        def get_pickled_cuts_funcobj(reader_instance, **kwargs):

            if 'cuts_funcobj' in kwargs.keys():
                if kwargs['cuts_funcobj'] == 'nocut':
                    cuts_funcobj = 'No cuts were applied: all rows of the original catalog were kept'
                    if reader_instance._cuts_description != 'nocut':
                        raise SyntaxError("\nThe store_processed_halocat method was supplied with "
                            "keyword argument cuts_funcobj = 'nocut',\n"
                            "but this is inconsistent with the input supplied to the reader_obj\n")
                else:
                    if not callable(kwargs['cuts_funcobj']):
                        raise TypeError("The input cuts_funcobj must be a callable function")
                    else:
                        cuts_funcobj = kwargs['cuts_funcobj']
                    if reader_instance._cuts_description != 'User-supplied cuts_funcobj':
                        raise SyntaxError("\nThe store_processed_halocat method was supplied with "
                            "a function object for the keyword argument cuts_funcobj,\n"
                            "but this is inconsistent with the input supplied to the reader_obj\n")

            else:
                reader_name = reader_instance.__class__.__name__
                cuts_funcobj = ("Halo catalog cuts were made using "
                    "the default_halocat_cut method of "+reader_name)
                if reader_instance._cuts_description != 'Default cut set by default_halocat_cut':
                    raise SyntaxError("\nThe store_processed_halocat method was not supplied with "
                        "a function object for the keyword argument cuts_funcobj.\n"
                        "This is inconsistent with the input supplied to the reader_obj\n")

            return cuts_funcobj

        ### Add metadata to the hdf5 file
        f = h5py.File(output_full_fname)

        # Function object used to cut the original halo catalog
        cuts_funcobj = get_pickled_cuts_funcobj(reader_obj, **kwargs)
        pickled_cuts_funcobj = pickle.dumps(cuts_funcobj, protocol = 0)
        f.attrs['halocat_exact_cuts'] = pickled_cuts_funcobj

        time_right_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['time_of_original_reduction'] = time_right_now

        f.attrs['original_data_source'] = reader_obj.halocat_obj.original_data_source

        for note_key, note in notes.iteritems():
            f.attrs[note_key] = note

        f.close()
        #################################

        return output_full_fname
 
    def closest_halocat(
        self, location, catalog_type, simname, halo_finder, input_redshift, 
        **kwargs):
        """ Search the cache directory for the closest snapshot matching the 
        input specs. 

        Parameters 
        ----------
        location : string 
            Specifies the web or disk location to search for halo catalogs. 
            Optional values for `location` are:

                *  `web`

                * `cache`

                * a full pathname such as `/full/path/to/my/personal/halocats/`. 

        catalog_type : string
            String giving the type of catalog. 
            Should be `halos`, or `raw_halos`. 

        simname : string
            Nickname of the simulation, e.g. `bolshoi`. 

        halo_finder : string
            Nickname of the halo-finder, e.g. `rockstar`. 

        input_redshift : float
            Desired redshift of the snapshot. 

        version_name : string, optional
            For cases where multiple versions of the same halo catalog 
            are stored in the cache, 
            a matching version name must be supplied to disambiguate. 

        Returns
        -------
        output_fname : string 
            Filename of the closest matching snapshot. 

        redshift : float 
            Value of the redshift of the snapshot
        """
        filename_list = self.available_snapshots(
            location, catalog_type, simname, halo_finder)
        if custom_len(filename_list) == 0:
            return None

        halocat_obj = get_halocat_obj(simname, halo_finder)
        result = halocat_obj.closest_halocat(filename_list, input_redshift, 
            **kwargs)
        if custom_len(result) == 0:
            print("No halo catalogs found in cache for simname = %s "
                " and halo-finder = %s" % (simname, halo_finder))
            return None

        if location=='web':
            if catalog_type == 'raw_halos':
                dirname = halocat_obj.raw_halocat_web_location
            elif (catalog_type == 'halos'):
                dirname = cache_config.processed_halocats_web_location(
                    simname=simname, halo_finder=halo_finder)
            else:
                raise IOError("For web locations, the only supported catalog_type are: "
                    "%s, %s, or %s" % ('raw_halos', 'halos', 'particles'))
        elif location=='cache':
            dirname = cache_config.get_catalogs_dir(
                catalog_type, simname=simname, halo_finder=halo_finder)
        else:
            dirname = location
            if not os.path.exists(dirname):
                raise IOError("The closest_halocat method was passed "
                    "the following explicit dirname:\n%s\n"
                    "This location must be a directory on your local disk, \n"
                    "but no such location was detected")

        absolute_fname = os.path.join(dirname, result[0])

        return absolute_fname, result[1]


    def all_halocats_in_cache(self, catalog_type, **kwargs):
        """ Return a list of all filenames (including absolute path) 
        of halo catalogs in the cache directory. 

        Parameters 
        ----------
        catalog_type : string
            String giving the type of catalog. 
            Should be `halos`, or `raw_halos`. 

        simname : string, optional 
            Nickname of the simulation, e.g. `bolshoi`. 
            If not None, only the simname cache subdirectory 
            will be searched. 

        halo_finder : string, optional 
            Nickname of the halo-finder, e.g. `rockstar`. 
            If not None, only the simname/halo_finder 
            cache subdirectory will be searched. 

        Returns 
        -------
        all_cached_files : list 
            List of strings containing the filenames. 

        """

        rootdir = cache_config.get_catalogs_dir(catalog_type, **kwargs)

        all_cached_files = []
        for path, dirlist, filelist in os.walk(rootdir):
            for fname in filelist:
                all_cached_files.append(os.path.join(path, fname))

        return all_cached_files

    def check_for_existing_halocat(self, location, catalog_type, 
        simname, halo_finder, **kwargs):
        """ Method searches the appropriate location in the 
        cache directory for the input fname, and returns a boolean for whether the 
        file is already in cache. 

        Parameters 
        ----------
        location : string 
            Specifies the web or disk location to search for halo catalogs. 
            Optional values for `location` are:

                * `cache`

                * pathname, either absolute or relative.  

        catalog_type : string
            String giving the type of catalog. 
            Should be `halos`, or `raw_halos`. 

        simname : string 
            Nickname of the simulation, e.g. `bolshoi`. 
            Must be specified if no `fname` keyword argument is given. 

        halo_finder : string 
            Nickname of the halo-finder, e.g. `rockstar`. 
            Must be specified if no `fname` keyword argument is given. 

        fname : string, optional 
            Filename (including absolute path) of the catalog being searched for. 

        redshift : float, optional 
            Redshift of the snapshot being searched for. 
            Must be specified if no `fname` keyword argument is given. 

        dz_tol : float, optional
            Tolerance value determining how close the requested redshift must be to 
            the closest available snapshot to be considered a match. Default value is 0.1. 

        Returns 
        -------
        is_in_cache : bool or string
            If no match is found, returns False. If a matching is found, 
            the filename (including absolute path) is returned. 
        """

        if location == 'cache':
            dirname = cache_config.get_catalogs_dir(catalog_type, 
                simname=simname, halo_finder=halo_finder)
        else:
            dirname = os.path.abspath(location)

        if 'fname' in kwargs.keys():
            fname = kwargs['fname']

            potential_matches = []
            for path, dirlist, filelist in os.walk(dirname):
                for f in filelist:
                    if path == dirname:
                        potential_matches.append(os.path.join(path, f))
            if fname[-3:] == '.gz':
                file_pattern = '*'+os.path.basename(fname[:-3])+'*'
            else:
                file_pattern = '*'+os.path.basename(fname)+'*'
            matching_catalogs = fnmatch.filter(potential_matches, file_pattern)

            if len(matching_catalogs) == 0:
                return False
            elif len(matching_catalogs) == 1:
                return os.path.abspath(matching_catalogs[0])
            elif len(matching_catalogs) == 2:
                length_matching_catalogs = [len(s) for s in matching_catalogs]
                idx_sorted = np.argsort(length_matching_catalogs)
                sorted_matching_catalogs = list(np.array(matching_catalogs)[idx_sorted])
                fname1, fname2 = sorted_matching_catalogs
                if fname2[:-3] == fname1:
                    warnings.warn("For filename:\n%s,\n"
                    "both the file and its uncompressed version appear "
                    "in the cache directory. " % fname2)
                return os.path.abspath(sorted_matching_catalogs[1])
            else:
                raise IOError("More than 1 matching catalog found in cache directory")

        else:
            if 'redshift' not in kwargs.keys():
                raise IOError("If the 'fname' keyword argument is not passed to "
                    "check_for_existing_halocat, then you must pass "
                    "the 'redshift' keyword argument")
            else:
                redshift = kwargs['redshift']
                if 'dz_tol' in kwargs.keys():
                    dz_tol = kwargs['dz_tol']
                else:
                    dz_tol = 0.1

        # Check whether there are existing catalogs matching the file pattern 
        # that is about to be downloaded
        closest_cat, closest_redshift = self.closest_halocat(
            location, catalog_type, simname, halo_finder, redshift)

        if abs(closest_redshift - redshift) > dz_tol:
            msg = (
                "No raw %s halo catalog has \na redshift within %.2f " + 
                "of the input_redshift = %.2f.\n The closest redshift for these catalogs is %.2f"
                )
            print(msg % (simname, dz_tol, redshift, closest_redshift))
            return False
        else:
            return os.path.abspath(closest_cat)


    def download_preprocessed_halo_catalog(self, simname, halo_finder, input_redshift, 
        **kwargs):
        """ Method to download one of the pre-processed binary files 
        storing a reduced halo catalog.  

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation, e.g. `bolshoi`. 

        halo_finder : string 
            Nickname of the halo-finder, e.g. `rockstar`. 

        input_redshift : float 
            Redshift of the requested snapshot. Must match one of the 
            available snapshots, or a prompt will be issued providing the nearest 
            available snapshots to choose from. 

        dz_tol : float, optional
            Tolerance value determining how close the requested redshift must be to 
            some available snapshot before issuing a warning. Default value is 0.1. 

        download_loc : string, optional
            Absolute pathname of where the raw halo catalog will be stored. 
            Default is the halotools cache directory. 

        overwrite : boolean, optional
            If a file with the same filename already exists 
            in the requested download location, the `overwrite` boolean determines 
            whether or not to overwrite the file. Default is False, in which case 
            no download will occur if a pre-existing file is detected. 

        Returns 
        -------
        output_fname : string  
            Filename (including absolute path) of the location of the downloaded 
            halo catalog.  
        """

        if HAS_SOUP == False:
            print("Must have BeautifulSoup installed to use Halotools Catalog Manager")
            return 

        if 'dz_tol' in kwargs.keys():
            dz_tol = kwargs['dz_tol']
        else:
            dz_tol = 0.1

        if 'overwrite' in kwargs.keys():
            overwrite = kwargs['overwrite']
        else:
            overwrite = False

        halocat_obj = get_halocat_obj(simname, halo_finder)
        list_of_available_snapshots = halocat_obj.preprocessed_halocats_available_for_download
        closest_snapshot_fname, redshift_of_closest_match = (
            halocat_obj.closest_halocat(
            list_of_available_snapshots, input_redshift)
            )

        if abs(redshift_of_closest_match - input_redshift) > dz_tol:
            msg = (
                "No raw %s halo catalog has \na redshift within %.2f " + 
                "of the input_redshift = %.2f.\n The closest redshift for these catalogs is %.2f"
                )
            print(msg % (simname, dz_tol, input_redshift, redshift_of_closest_match))
            return 

        webloc = os.path.join(
            os.path.join(
                sim_defaults.processed_halocats_webloc, simname), halo_finder)
        url = os.path.join(webloc, closest_snapshot_fname)

        if 'download_loc' in kwargs.keys():
            download_loc = kwargs['download_loc']
            # We were given an explicit path to store the catalog
            # Check that this path actually exists, and if so, use it 
            if not os.path.exists(download_loc):
                raise IOError("Input directory name %s for download location"
                    "of raw halo catalog does not exist" % download_loc)
            else:
                output_fname = os.path.join(download_loc, closest_snapshot_fname)
        else:
            # We were not given an explicit path, so use the default Halotools cache dir
            cache_dirname = cache_config.get_catalogs_dir('halos', 
                simname=simname, halo_finder=halo_finder)
            output_fname = os.path.join(cache_dirname, closest_snapshot_fname)
            download_loc = 'cache'

        # Check whether there are existing catalogs matching the file pattern 
        # that is about to be downloaded
        is_in_cache = self.check_for_existing_halocat(
            download_loc, 'halos', simname, halo_finder, 
            fname=os.path.basename(closest_snapshot_fname))

        if is_in_cache != False:
            if overwrite == True:
                warnings.warn("Downloading halo catalog and overwriting existing file %s" % output_fname)
            else:
                msg = ("The following filename already exists in your cache directory: \n\n%s\n\n"
                    "If you really want to overwrite the file, \n"
                    "you must call the same function again \n"
                    "with the keyword argument `overwrite` set to `True`")
                print(msg % output_fname)
                return None

        else:
            start = time()
            download_file_from_url(url, output_fname)
            end = time()
            runtime = (end - start)/60.
            print("\nTotal runtime to download snapshot = %.1f minutes\n" % runtime)
            return output_fname

    def load_halo_catalog(self, verbose=False, **kwargs):
        """ Method returns an Astropy Table object of halos 
        that have been stored as a processed binary hdf5 file. 

        Parameters 
        ----------
        fname : string, optional 
            Filename (including absolute path) where the hdf5 file is stored. 

        simname : string, optional 
            Nickname of the simulation, e.g. `bolshoi`. 

        halo_finder : string, optional 
            Nickname of the halo-finder, e.g. `rockstar`. 

        redshift : float, optional 
            Redshift of the desired snapshot. 

        verbose : bool, optional 
            If True, a range of print statements will be issued as sanity checks 
            on bookkeeping. Default is False. 

        Returns 
        -------
        t : table 
            Astropy Table object storing the halo catalog data. 
        """

        if 'fname' in kwargs.keys():
            if verbose is True:
                print("Loading halo catalog "
                    "with the following absolute path: \n%s\n" % kwargs['fname'])
            return Table.read(kwargs['fname'], path='halos')
        else:
            simname = kwargs['simname']
            halo_finder = kwargs['halo_finder']
            redshift = kwargs['redshift']

            result = self.closest_halocat(
                'cache', 'halos', simname, halo_finder, redshift)
            if custom_len(result) == 0:
                return None
            else:
                fname, z = result[0], result[1]
                if verbose is True:
                    print("Loading z = %.2f halo catalog "
                        "with the following absolute path: \n%s\n" % (z, fname))
                return Table.read(fname, path='halos')

    def download_all_default_catalogs(self):
        """ Convenience method used to download all pre-processed halo catalogs 
        that are not already in the cache directory.

        Returns 
        -------
        new_downloads : list 
            List of strings of all newly-downloaded filenames. 
        """
        location = 'web'
        catalog_type = 'halos'

        new_downloads = []
        halocat_list = self.available_halocats
        for simname, halo_finder in halocat_list:
            halocat_obj = get_halocat_obj(simname, halo_finder)
            urls = self.available_snapshots(location, catalog_type, simname, halo_finder)
            for url in urls:
                fname = os.path.basename(url)
                scale_factor_substr = halocat_obj.get_scale_factor_substring(fname)
                a = float(scale_factor_substr)
                z = (1/a) - 1
                result = self.download_preprocessed_halo_catalog(simname, halo_finder, z)
                if result is not None:
                    new_downloads.append(result)

        return new_downloads


###################################################################################################
class RockstarReader(object):
    """ Class containing methods used to read raw ASCII data of Rockstar hlist files. 

    Each new raw halo catalog must be processed with its own instance of this class. 
    """

    def __init__(self, input_fname, simname, halo_finder, **kwargs):
        """
        Parameters 
        -----------
        input_fname : string 
            Name of the file (including absolute path) to be processed. 

        simname : string 
            Nickname of the simulation, e.g. `bolshoi`. 

        halo_finder : string 
            Nickname of the halo-finder, e.g. `rockstar`. 

        cuts_funcobj : function object, optional
            Function used to apply cuts to the rows of the ASCII data. 
            `cuts_funcobj` should accept a structured array as input, 
            and return a boolean array of the same length. 
            If None, default cut is set by `default_halocat_cut`. 
            If set to the string ``nocut``, all rows will be kept. 
            The `cuts_funcobj` must be a callable function defined 
            within the namespace of the `RockstarReader` instance, and 
            it must be a stand-alone function, not a bound method of 
            some other class.  
        """

        if not os.path.isfile(input_fname):
            if not os.path.isfile(input_fname[:-3]):
                raise IOError("Input filename %s is not a file" % input_fname)
                
        self.fname = input_fname
        self._uncompress_ascii()
        self.simname = simname
        self.halo_finder = halo_finder
        self.halocat_obj = get_halocat_obj(simname, halo_finder)

        if 'cuts_funcobj' in kwargs.keys():
            if kwargs['cuts_funcobj'] == 'nocut':
                g = lambda x : np.ones(len(x), dtype=bool)
                self.cuts_funcobj = g
                self._cuts_description = 'nocut'
            else:
                if callable(kwargs['cuts_funcobj']):
                    self.cuts_funcobj = kwargs['cuts_funcobj']
                    self._cuts_description = 'User-supplied cuts_funcobj'
                else:
                    raise TypeError("The input cuts_funcobj must be a callable function")
                    
        else:
            self.cuts_funcobj = self.default_halocat_cut
            self._cuts_description = 'Default cut set by default_halocat_cut'

    def default_halocat_cut(self, x):
        """ Function used to provide a simple cut on a raw halo catalog, 
        such that only rows with :math:`M_{\\rm peak} > 300m_{\\rm p}` 
        pass the cut. 

        Parameters 
        ----------
        x : array 
            Length-Nhalos structured numpy array, presumed to have a field called `mpeak`. 

        Returns 
        -------
        result : array
            Length-Nhalos boolean array serving as a mask. 
        """

        return x['mpeak'] > ( 
            self.halocat_obj.simulation.particle_mass*
            sim_defaults.Num_ptcl_requirement)

    def file_len(self):
        """ Compute the number of all rows in the raw halo catalog. 

        Parameters 
        ----------
        fname : string 

        Returns 
        -------
        Nrows : int
     
        """
        with open(self.fname) as f:
            for i, l in enumerate(f):
                pass
        Nrows = i + 1
        return Nrows

    def header_len(self,header_char='#'):
        """ Compute the number of header rows in the raw halo catalog. 

        Parameters 
        ----------
        fname : string 

        header_char : str, optional
            string to be interpreted as a header line

        Returns 
        -------
        Nheader : int

        Notes 
        -----
        All empty lines that appear in header 
        will be included in the count. 

        """
        Nheader = 0
        with open(self.fname) as f:
            for i, l in enumerate(f):
                if ( (l[0:len(header_char)]==header_char) or (l=="\n") ):
                    Nheader += 1
                else:
                    break

        return Nheader

    def get_header(self, Nrows_header_total=None):
        """ Return the header as a list of strings, 
        one entry per header row. 

        Parameters 
        ----------
        fname : string 

        Nrows_header_total :  int, optional
            If the total number of header rows is not known in advance, 
            method will call `header_len` to determine Nrows_header_total. 

        Notes 
        -----
        Empty lines will be included in the returned header. 

        """

        if Nrows_header_total is None:
            Nrows_header_total = self.header_len(self.fname)

        print("Reading the first %i lines of the ascii file" % Nrows_header_total)

        output = []
        with open(self.fname) as f:
            for i in range(Nrows_header_total):
                line = f.readline().strip()
                output.append(line)

        return output

    def _uncompress_ascii(self):
        """ If the input fname has file extension `.gz`, 
        then the method uses `gunzip` to decompress it, 
        and returns the input fname truncated to exclude 
        the `.gz` extension. If the input fname does not 
        end in `.gz`, method does nothing besides return 
        the input fname. 
        """
        if self.fname[-3:]=='.gz':
            print("...uncompressing ASCII data")
            os.system("gunzip "+self.fname)
            self.fname = self.fname[:-3]
        else:
            pass

    def _compress_ascii(self):
        """ Recompresses the halo catalog ascii data, 
        and returns the input filename appended with `.gz`.  
        """
        if self.fname[-3:]!='.gz':
            print("...re-compressing ASCII data")
            os.system("gzip "+self.fname)
            self.fname = self.fname + '.gz'
        else:
            pass


    def read_halocat(self, **kwargs):
        """ Reads the raw halo catalog in chunks and returns a structured array
        after applying cuts.

        Parameters 
        ----------
        cuts_funcobj : function object, optional keyword argument
            Function used to determine whether a row of the raw 
            halo catalog is included in the reduced binary. 
            Input of the `cuts_funcobj1 must be a structured array 
            with some subset of the field names of the 
            halo catalog. Output of the `cuts_funcobj` must 
            be a boolean array of length equal to the length of the 
            input structured array. 
            Default is set by the `default_halocat_cut` method. 

        nchunks : int, optional keyword argument
            `read_halocat` reads and processes ascii 
            in chunks at a time, both to improve performance and 
            so that the entire raw halo catalog need not fit in memory 
            in order to process it. The total number of chunks to use 
            can be specified with the `nchunks` argument. Default is 1000. 

        """
        start = time()

        if 'nchunks' in kwargs.keys():
            Nchunks = kwargs['nchunks']
        else:
            Nchunks = 1000

        dt = self.halocat_obj.halocat_column_info

        file_length = self.file_len()
        header_length = self.header_len()
        chunksize = file_length / Nchunks
        if chunksize == 0:
            chunksize = file_length # data will now never be chunked
            Nchunks = 1


        print("\n...Processing ASCII data of file: \n%s\n " % self.fname)
        print(" Total number of rows in file = %i" % file_length)
        print(" Number of rows in detected header = %i \n" % header_length)
        if Nchunks==1:
            print("Reading catalog in a single chunk of size %i\n" % chunksize)
        else:
            print("...Reading catalog in %i chunks, each with %i rows\n" % (Nchunks, chunksize))

        chunk_counter = 0
        chunk = []
        container = []
        iout = np.round(Nchunks / 10.).astype(int)
        for linenum, line in enumerate(open(self.fname)):
            if line[0] == '#':
                pass
            else:
                parsed_line = line.strip().split()
                chunk.append(tuple(parsed_line))  
        
            if (linenum % chunksize == 0) & (linenum > 0):

                chunk_counter += 1
                if (chunk_counter % iout)==0:
                    print("... working on chunk # %i of %i\n" % (chunk_counter, Nchunks))

                try:
                    a = np.array(chunk, dtype = dt)
                except ValueError:
                    Nfields = len(dt.fields)
                    print("Number of fields in np.dtype = %i" % Nfields)
                    Ncolumns = []
                    for elt in chunk:
                        if len(elt) not in Ncolumns:
                            Ncolumns.append(len(elt))
                    print("Number of columns in chunk = ")
                    for ncols in Ncolumns:
                        print ncols
                    print chunk[-1]
                    raise ValueError("Number of columns does not match length of dtype")

                container.append(a[self.cuts_funcobj(a)])
                chunk = []

        a = np.array(chunk, dtype = dt)
        container.append(a[self.cuts_funcobj(a)])

    # Bundle up all array chunks into a single array
        for chunk in container:
            try:
                output = np.append(output, chunk)
            except NameError:
                output = np.array(chunk) 
                

        end = time()
        runtime = (end-start)
        if runtime > 60:
            runtime = runtime/60.
            msg = "Total runtime to read in ASCII = %.1f minutes\n"
        else:
            msg = "Total runtime to read in ASCII = %.1f seconds\n"
        print(msg % runtime)

        return output





###################################################################################################

