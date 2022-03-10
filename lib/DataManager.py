from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
import shutil

class DataManager():
    '''
    The class is used to manage MD simulations database for BO FF parameterization.
    BO uses a mesh, which nodes correspond to different values of FF parameters of interest.
    "Nodal" FF parameters are used to run MD simulations. 
    DataManager:
     -- create a structure where each node
    
    
    Attributes
    ----------
    bounds : dict
        {resid: {param_name : [x_min, x_max]}}, resid and parameter names as specified in FF file
    df_ff : pandas.DataFrame
        df includes parameter value for each resid and parameter
    ff_fpath : pathlib.Path
        path to FF file
    ff_start: int
        line index to start reading FF file
    ff_end: int
        line index to stop reading FF file    
    grid_spec: dictionary
        a dict to setup search space, where array [x_min, x_max, N_vertices] is assigned to each parameter,
        e.g.: {"Li" : {"sigma" : [0., 0.5, 1.]}}
    ndim : int
        number of dimensions (number of FF parameters to tune)  
    nodes : dict
        { resid : {param_name : node_coodrs}}, node_coodrs - list of floats
    node_fpath : pathlib.Path
        path to grid configuration json file 
        md_templ_path : pathlib.Path instance
        path to md template with simulation inputs and bash-executive files    
    md_templ_path : pathlib.Path instance
        path to md template with simulation inputs and bash-executive files    
    params - list
        a sorted list of "resid parameter_name" pairs
    root : pathlib.Path instance
        path to database dir

    Methods
    -------
    casedir_exists(mutliidx)
        checks if directory with the name according to given multiidx exists
    check_inclusion(pdict)
        checks if point pdict is inside of the bounds
    convert_to_pdict(plist)
        converts coordinates of a point written as a dict into coordinates as a list
    convert_to_plist(pdict)
        converts coordinates of a point written as a list into coordinates as a dict
    create_case(pdict)
        creates a folder named according to multiindex, modifies FF file in the folder
    get_multiidx(pdict)
        returns multiindex of the gride node closest to pdict
    get_node_dict():
        returns a dictionary with node coordinates
    make_casename(pdict)
        returns str - casename according to pdict
    _check_node_f(node_fpath):
        method check if grid_spec from a given json file  is equivalent to class grid_spec
    _read_node_f(node_fpath):
        returns a dict read from a json file with node coordinates
    _read_ff_file(ff_fpath):
        reads specified ff file as a pandas DataFrame
    _save_node_f():
        writes dict with node coordinates self.nodes as a json file
    _save_ff_file(self, df, ff_fpath_out):
        writes ff parameters into a given ff file 
    '''
    def __init__(self, root, md_templ, grid_spec, ff_fname, ff_start, ff_end, node_fname='_nodes.json'):
        """
        Parameters
        ----------
        root - str
            a path to the database dir. If doesn't exist, it is to be created.
        md_templ - str
            a path to MD template folder with prepared simulation data and executable bash script
        grid_spec - dict
            a dict resid: {parameterX : [x_min, x_max, Nx]} , Nx - number of nodes for parameter x = x1 ... xN)
        ff_fname - str
            a file name of the file with specified force field parameters
        ff_start - int
            lines from ff_start to ff_end of file ff_fname are read as pandas.DataFrame
        ff_end - int
            lines from ff_start to ff_end of file ff_fname are read as pandas.DataFrame
        node_fname - str
            file name to save node coordinates for specified parametera
        """
        if not isinstance(root, Path):
            root = Path(root)
        self.root = root

        self.md_templ = md_templ

        self.grid_spec = grid_spec
        params_ = []
        for resid in self.grid_spec:
            for param_name in self.grid_spec[resid]:
                params_.append((resid, param_name))
        params_ = sorted(params_)
        self.params = tuple(params_)
        self.ndim = len(self.params)

        bounds_, nodes_ = [], []
        for resid, param_name in self.params:
            x_min, x_max, Nx = self.grid_spec[resid][param_name]
            bounds_.append((x_min, x_max))
            nodes_.append(np.linspace(x_min, x_max, Nx))
        self.bounds =tuple(bounds_)
        self.nodes = tuple(nodes_)


        self.ff_fpath = self.md_templ/ff_fname
        self.ff_start = ff_start
        self.ff_end = ff_end
        self.df_ff =self._read_ff_file(self.ff_fpath)

        self.node_fpath = self.root/node_fname

        print(f"\t Checking {node_fname} file...", end="")
        if os.path.exists(self.node_fpath):
            print("Exists! ... ", end="")
            try:
                self._check_node_f(self.node_fpath)
            except:
                er_mes = f"{self.node_fpath} does not correspond to grid_spec!" + \
                    f"Check {self.root/node_fname}"
                print(er_mes)
                raise ValueError(0, er_mes)
            else:
                print("success!")
        else:
            print(f"Not found! Creating {node_fname} according to grid_spec ... ", end="")
            try:
                self._save_node_f()
            except:
                er_mes = f"could not create {self.node_fpath}"
                print(er_mes)
                raise OSError(0, er_mes)
            else:
                print("success!") 

        
        print(f"Initialization: database {self.root} ... ", end="")
        if os.path.exists(self.node_fpath):
            print(f"already exists! Checking  {node_fname} file ....")
            self._check_node_f(self.node_fpath)
        print("Success! Database manager is created!!!")

    def casedir_exists(self, multiidx):
        """
        check if there is a directory with the name specified by multiidx
        """
        casename = self.make_casename(multiidx)
        casepath = self.root/casename
        return os.path.exists(casepath)
    
    def check_inclusion(self, pdict):
        """
        checks if point with coordinates pdict = {resid : {param_name : param_value}} 
        is inside of the search space defined by self.grid_spec
         """
        plist = self.convert_to_plist(pdict)
        for i in range(len(plist)):
            resid, param_name = self.params[i]
            xi = plist[i]
            low_b, high_b = self.bounds[i]
            assert (xi  >= low_b and xi <= high_b), "Point out of the bound ({resid}, {param_name})!"
        return True

    def convert_to_pdict(self, plist):
        """
        converts point cordinates as dict (pdict) to 
        point coordinates as list (plist) according to self.params order        
        """
        pdict = {}
        for i, (resid, param_name) in enumerate(self.params):
            param_value = plist[i]
            pdict[resid] = pdict[resid] if resid in pdict else {}
            pdict[resid].update({param_name: param_value})
        return pdict

    def convert_to_plist(self, pdict):
        """
        converts point cordinates as list (pdict) to point coordinates 
        as dict (plist) according to self.params order        
        """
        plist = []
        for resid, param_name in self.params:
            param_value = pdict[resid][param_name]
            plist.append(param_value)
        return plist 

    def create_case(self, pdict):
        """
        Creates a folder with name corresponding to pdict. If folder exists,
        throws a warning. 
        Parameters:
        -----------
        pdict : dict
            dict of paramater values - coordinates, {resid : {param_name : param_value}} 
        Returns:
        --------
            path to casedir as str / None if casedir wasn't created
        """
        multiidx = self.get_multiidx(pdict)
        if not multiidx:
            return False

        casename = self.make_casename(multiidx)
        ff_fname = self.ff_fpath.name
        casepath = self.root/casename

        if not self.check_inclusion(pdict):
            print("RANGE ERROR!")
            return None
        if self.casedir_exists(multiidx):
            print(f"DataManager Warning! The casedir {casename} already exists,",
             "no changes are made.",
             "The existing MD configuration will be used")
            return casepath
        else:
            shutil.copytree(self.md_templ, self.root/casename)

        df_ff_new = self.df_ff.copy()
        for i, (resid, param_name) in enumerate(self.params):
            idx = multiidx[i]
            param_val = self.nodes[i][idx]
            df_ff_new.at[resid, param_name] = '{:6.5e}'.format(param_val)

        ff_fpath_new = casepath/ff_fname
        self._save_ff_file(df_ff_new, ff_fpath_new)
        return casepath
        
    def get_multiidx(self, pdict):
        """
        checks inclusion of point pdict and assign multiindex of the nearest
        vertices if point included
        WARNING! It is assumed that a point is located near one of vertices. 
        The accuracy of assignments  of points in between is not guaranteed.     
        Parameters
        ----------
        pdict : dictionary
            dict of paramater values - coordinates, {resid : {param_name : param_value}}  
        Returns
        -------
        tuple / None
            returns tuple of int-indices or None if point doesn't belong to the search space
        """
        if not self.check_inclusion(pdict):
            return None
        
        mutliidx = []
        for i, (resid, par) in enumerate(self.params):
            nodes_i = self.nodes[i]
            nodes_i = np.array(nodes_i)
            xi = pdict[resid][par]
            idx = np.abs(nodes_i-xi).argmin()
            if not np.isclose(nodes_i[idx], xi):
                print("WARNING: get_multiidx: nodes coordinates don't match with point coordinates.", 
                "However, the closest multiidx is found and ff parameters",
                "are set according to grid node at this multiidx ")
            mutliidx.append(idx)
        return tuple(mutliidx)
    
    def get_node_dict(self):
        """
        Returns a dictionary with node coordinates for each resid and parameter, 
            {resid : {param_name : node_coords}}
        """   
        node_dict = {}
        for resid in self.grid_spec:
            node_dict[resid] = {}
            for param_name in self.grid_spec[resid]:
                x_min, x_max, Nx = self.grid_spec[resid][param_name]
                node_coord_i = np.linspace(x_min, x_max, Nx)
                node_coord_i = list(node_coord_i) # json doesn't save np.arrays as entries
                node_dict[resid][param_name] = node_coord_i
        return node_dict

    def make_casename(self, multiidx):
        """
        creates case name according to specified multiidx
        Parameters:
        -----------
        multiidx - list
            list of idx for each resid and parameter
        Returns:
        str as IDX1_IDX2_.._IDXN
        """
        casename = f'{multiidx[0] :03d}' 
        for idx in multiidx[1:]:
            casename += f'_{idx :03d}'
        return casename  

    def _check_node_f(self, node_fpath):
        """
        method check if grid_spec from a given json file  is equivalent to self.grid_spec
        Parameters
        ----------
        node_fpath - str
            a filepath to a file with written node coordinates for each parameter
        """
        node_coord_ext = self._read_node_f(node_fpath)

        ndim_ext = 0
        for i, (resid, param_name) in enumerate(self.params):
            coord_i = self.nodes[i]
            coord_i_ext = node_coord_ext[resid][param_name]
            if not np.allclose(coord_i, coord_i_ext):
                er_mes = f"_check_node_f: nodes coordinates do not match with {node_fpath}!"
                print(er_mes)
                raise ValueError(0, er_mes)
            ndim_ext += 1

        if ndim_ext != self.ndim:
            er_mes = f"_check_node_f: dimension dismatch! Check {node_fpath}!"
            print(er_mes)
            raise KeyError(0, er_mes)
        return True

    def _read_ff_file(self, ff_fpath):
        """
        Reads specified ff file as a pandas DataFrame
        """
        with open(ff_fpath, 'r',) as f:
            lines = f.readlines()
            df = pd.DataFrame(data=[l.split()[1:] for l in lines[self.ff_start:self.ff_end][1:]])
            col_names = lines[self.ff_start][1:]
            col_names = col_names.split()[1:]
            col_ind = list(range(df.shape[1]))
            col_dict = dict(zip(col_ind, col_names))
            
            row_names = [l.split()[0] for l in lines[self.ff_start:self.ff_end][1:]]
            row_ind = list(range(df.shape[0]))
            row_dict = dict(zip(row_ind, row_names))
            df = df.rename(columns=col_dict, index=row_dict)
            return df
    
    def _read_node_f(self, node_fpath):
        """
        reads json file with node coordinates as a dict
        """
        with open(node_fpath) as node_f:
            node_dict = json.loads(node_f.read())
        return node_dict

    def _save_ff_file(self, df, ff_fpath_out):
        """
        writes ff parameters as a Pandas DataFrame into a specified ff file
        Parameters:
        ----------
        df - pandas.DataFrame
            dataframe with specified resid, parameter names and values
        ff_fpath_out - str
            str to ff file to write ff parameters
        """
        with open(ff_fpath_out, 'r',) as f:
            lines = f.readlines()
        colums = df.columns
        for i in range(self.ff_start, self.ff_start+df.shape[0]):
            l = list(df.iloc[i-self.ff_start])
            l = [df.index[i-self.ff_start]] + l
            form = ['', '\t', '\t', '\t', '\t', '\t', '\t']
            l = [item for sublist in zip(form,l) for item in sublist]
            l = ''.join(l)
            lines[i+1] = l + '\n'
 
        f_out = open(ff_fpath_out, 'w')
        for l in lines:
            f_out.write(l)
        f_out.close()    
    
    def _save_node_f(self):
        """
        writes dict with node coordinates self.nodes as a json file
        (only if the file does not exist)
        Returns
        -------
            True if data are successully saved, False otherwise
        """
        if self.node_fpath.exists():
            print("File already exists!")
            return False

        node_dict = self.get_node_dict()
        print(node_dict)
        with open(self.node_fpath, "w") as node_f:
            json.dump(node_dict, node_f)
        return True
    
