import numpy as np
import pandas as pd
import pathlib as pl
import warnings

from scipy.interpolate import UnivariateSpline

class Sp8:
    
    def __init__(
        self,
        dir,
        *args,
        data_list=None,
        E_shift=0,
        **kwargs
    ):
        # if import_kwargs is None:
        #     import_kwargs = {}
        if data_list is not None:
            self.data_list = data_list
        else:
            self.path_generator(dir, *args, **kwargs)
        self.arr_list = []
        for f in self.data_list:
            self.arr_list.append(self.import_data(f, E_shift=E_shift))
    
    def path_generator(
        self,
        dir,
        suffix,
        runs,
        scans=None,
        ext='.dat',
        run_format='04',
        scan_format='03',
        exclude_scans=None,
        run_scan_sep='_'
    ):
        '''
        Returns a list of data file paths (pl.Path() instances).
        '''
        if exclude_scans is None:
            exclude_scans = ()
        
        self.dir = pl.Path(dir)
        # glob everything
        self.data_list = sorted(self.dir.glob(f"{suffix}*{ext}"))
        # filter runs
        # run ID string formatting
        runs = [format(r, run_format) for r in runs]
        self.data_list = [d for d in self.data_list if d.stem[(-int(run_format)-int(scan_format)-len(run_scan_sep)):(-int(scan_format)-len(run_scan_sep))] in runs]
        if scans is not None:
            if isinstance(scans, int):
                scans = [s for s in range(1, scans+1)]
            # scan ID string formatting
            scans = [format(s, scan_format) for s in scans if s not in exclude_scans]
            self.data_list = [d for d in self.data_list if d.stem[-int(scan_format):] in scans]
        return self.data_list
    
    def import_data(
        self,
        path,
        data_list=None,
        header=None,
        left_str="D=",
        right_str="A",
        E_shift=0,
        **kwargs
    ):
        '''
        Reads a raw SPring-8 data file and outputs an array of mu vs. E.
        '''
        if data_list is not None:
            self.data_list = data_list
            
        df = pd.read_csv(
            path,
            header=header,
            **kwargs
        )
        
        # get monochromator d-spacing
        str = df[df.iloc[:, 0].str.contains('D=')].iloc[0, 0]
        d_spacing = float(str[str.index(left_str)+len(left_str):str.index(right_str)])
        
        # find start of data
        skiprows = df[df.iloc[:, 0].str.contains('Offset')].index.values[0] + 1
        
        # slice and dice
        raw_arr = df.iloc[skiprows:, 0].str.split().apply(pd.to_numeric).apply(pd.Series).to_numpy()
        arr = np.empty((len(raw_arr), 2))
        # return mu vs E
        arr[:, 0] = self.energy(raw_arr[:, 1], d_spacing) + E_shift
        arr[:, 1] = -np.log(raw_arr[:, -1]/raw_arr[:, -2])
        arr = arr[arr[:, 0].argsort(), :]
        return arr
    
    def interpolate_and_average(
        self,
        bounds=None, 
        step=None,
        grid_points=None,
        s=0.0001
    ):
        if grid_points is None:
            grid_points = np.arange(bounds[0], bounds[1], step)
        self.E_out = np.empty((len(grid_points), len(self.arr_list)))
        self.arr_avg = np.empty((len(grid_points), 2))
        for i, arr in enumerate(self.arr_list):
            self.E_out[:, i] = self.interpolate_and_resample(
                arr, bounds=bounds, step=step, grid_points=grid_points, s=s
            )[:, 1]
        self.arr_avg[:, 0] = grid_points
        self.arr_avg[:, 1] = np.average(self.E_out, axis=1)
        self.E_max = self.arr_avg[self.arr_avg[:, 1].argmax(), 0]
    
    @classmethod
    def energy(cls, theta, d, n=1):
        '''
        Calculates energy from monochromator orientation and angle using Bragg's law.
        '''
        return 12398 / cls.bragg(theta, d, n)
    
    @staticmethod
    def bragg(theta, d, n=1):
        return n * 2 * d * np.sin(np.radians(theta))
    
    @staticmethod
    def interpolate_and_resample(
        arr, 
        bounds=None, 
        step=None,
        grid_points=None,
        s=0.0001
    ):
        """
        Use splines to reevaluate a DataFrame on a specified grid for a given column.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to reevaluate.
        column (str): The column name to base the reevaluation on.
        grid_points (array-like): The grid points to reevaluate the DataFrame on.
        
        Returns:
        pd.DataFrame: The reevaluated DataFrame.
        """
        if grid_points is None:
            if bounds is None:
                try:
                    bounds = (min(arr[:, 0]), max(arr[:, 1]))
                except:
                    warnings.warn('Auto-detection of x-bounds failed.')
            grid_points = np.arange(bounds[0], bounds[1], step)
        
        arr_out = np.empty((len(grid_points), 2))
        spline = UnivariateSpline(arr[:, 0], arr[:, 1], s=s)
        arr_out[:, 0] = grid_points
        arr_out[:, 1] = spline(grid_points)
        return arr_out
    
    @staticmethod
    def make_grid(
        e0,
        emin,
        emax,
        step=0.1,
        pre_stop=50,
        pre_step=1,
        post_start=None,
        post_step=None
    ):
        if post_step is None:
            post_step = step
        grid = np.arange(emin, e0-pre_stop, pre_step)
        if post_start is None:
            grid = np.append(grid, np.arange(e0-pre_stop, emax, step))
        else:
            grid = np.append(grid, np.arange(e0-pre_stop, e0+post_start, step))
            grid = np.arange(e0+post_start, emax, post_step)
        return grid