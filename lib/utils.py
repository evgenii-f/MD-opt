import os, subprocess
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

def run_md_case(case_dirpath, run_cmd, force=False, verbose=True):
    """
    Function runs a given console command run_cmd from a given derictory.
    It is aimed to run GROMACS/LAMMPS simulations by passing right commands.

    Parameters:
    -----------
    casedir_path : str
        path to directory with simulation data
    run_cmd : str
        console command to run
    force : bool
        defines behaviour if output .xvg file are found in casedir. 
        True - run simulation anyway, False - exit and return False
    verbose : bool
        if False, warning messages are suppressed 
    Returns:
    -------
        True if command is executed successfully, False - otherwise

    """
    xvg_files = find_xvg_files(case_dirpath)
    if len(xvg_files) > 0 and not force:
        if verbose:
            print(f"run_md_case Warning: xvg files are found in {case_dirpath}! ",
                   "To redo simulation, choose 'force' mode")
        return False
    cwd = os.getcwd()
    os.chdir(case_dirpath)
    os.system(run_cmd)
    os.chdir(cwd)   
    return True

def compute_total_cost(case_dirpath, ref_dirpath, weight_map,
                       x_max=None, smooth=True, p_norm=2, headers=(0, 0)):
    if not isinstance(case_dirpath, Path):
        case_dirpath = Path(case_dirpath)
    l_total = 0
    for atom_pair, weight in weight_map.items():
        rdfpath = case_dirpath/f'rdf_{atom_pair}.xvg'
        refpath = ref_dirpath/f'rdf_{atom_pair}.xvg'
        l = compute_cost(rdfpath, refpath, x_max, smooth, p_norm, headers)
        l_total += l * weight
    return l_total

def compute_cost(rdfpath, refpath, x_max=None, smooth=True, p_norm=2, headers=(0, 0)):
    """
    computes cost as a p-norm of difference between given vector and 
    reference vector (RDFs). Preliminary checks if distribution samples
    were taken from the same x-points
    Parameters
    ---------
    rdfpath - str
        path to xvg file with a given radial distribution
    refpath - str
        path to xvg file with reference radial distribution
    x_max - float/None
        maximum distance to compute RDF. If not - defined by given distribution
    smooth - bool
        defines if distrubution are smoothered with polynomial
    p_norm - int
        p-norm
    headers - [header1, header2]
        number of lines to skip in xvg files of given and reference distributions
        correspondingly
    Returns
        Computed cost value as a float
    """
    arr = np.genfromtxt(rdfpath, skip_header=headers[0])
    x, g = arr[:, 0], arr[:, 1]
    g = g / g.sum()

    arr_ref= np.genfromtxt(refpath, skip_header=headers[1])
    x_ref, g_ref = arr_ref[:, 0], arr_ref[:, 1]
    g_ref = g_ref / g_ref.sum()

    if x_max:
        ind_last = np.where(x > x_max)[0][0]
        x, g = x[:ind_last], g[:ind_last]

        ind_last_ref = np.where(x_ref > x_max)[0][0]
        x_ref, g_ref = x_ref[:ind_last_ref], g_ref[:ind_last_ref]
    
    assert np.allclose(x, x_ref), f"x and x_ref mismatch! Check RDF in {rdfpath} and {refpath}"

    diff = g - g_ref
    l = np.linalg.norm(diff, p_norm) / (np.abs(g_ref)**p_norm).sum()
    return l

def find_xvg_files(dirpath):
    """
    Returns list of file names with .xvg extension in a given directory
    """
    files = os.listdir(dirpath)
    xvg_files = [f for f in files if f[-4:] == '.xvg']

    return xvg_files

def plot2D_BO(X1, X2, X_used, mean, variance, alpha_full, 
              plist_new=None, plist_target=None, ax_labels=['par 1', 'par 2'],
              gen_title=None, s=100, fs=12, fs_ax=12, cmap='inferno', 
              alpha=0.75, savepath=None, mean_lim = (0, 2.5),
              annotations=None, 
             ):
    """
    Plots 2D contours of mean, variance and acquisition function,
    and 3D contour of mean
    """
    
    X_grid = np.meshgrid(X1, X2)
    X_grid = np.array(X_grid)
    X_grid = X_grid.T.reshape(-1, 2)
    X1_grid = X1
    X2_grid = X2
    
    idx_min = np.argmin(mean)
    x1_min, x2_min = X_grid[idx_min]
    
    mean_ = mean.reshape(len(X1), len(X2)).T
    variance_ = variance.reshape(len(X1), len(X2)).T
    alpha_full_ = alpha_full.reshape(len(X1), len(X2)).T
    
    if not plist_new is None:
        x1_new, x2_new = plist_new
    if not plist_target is None:
        x1_target, x2_target = plist_target
        
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    titles = ['Contour Predicted Mean', 'Contour Variance', 'Contour EI']
    Fs = [mean_, variance_, alpha_full_]
    for ax_i, title, f in zip((1, 2, 4), titles, Fs):
        ax = fig.add_subplot(2, 2, ax_i)
        im = ax.contourf(X1_grid, X2_grid, f, cmap=cmap, alpha=alpha, filled=True)
        ax.scatter(X_used[:, 0], X_used[:, 1], clip_on=False)
        if annotations:
            for i, ann in enumerate(annotations):
                ax.annotate(ann, X_used[i], fontsize=fs-2)
        ax.scatter(x1_min, x2_min, c='black', s=s, marker='x', clip_on=False, label='predicted min' if ax_i==1 else None)
        ax.scatter(X_used[:, 0], X_used[:, 1], clip_on=False, label = 'utilized dp'if ax_i==1 else None)
        if not plist_target is None:
            ax.scatter(x1_target, x2_target, marker='x', s=s, c ='r', clip_on=False, label = 'given ff' if ax_i==1 else None)
        if not plist_new is None:  
            ax.scatter(x1_new, x2_new, marker='^', s=s, c ='m', clip_on=False, label = 'next step' if ax_i==1 else None)
        ax.set_title(title, fontsize=fs)
        ax.set_xlabel(ax_labels[0], fontsize=fs)
        ax.set_ylabel(ax_labels[1], fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs_ax)
        ax.grid()
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=fs_ax)
    
    ax2 = fig.add_subplot(2, 2, 3, projection='3d')

    X1_, X2_ = np.meshgrid(X1, X2)

    im2 = ax2.plot_surface(X1_, X2_, mean_, cmap=cmap, alpha=alpha)
    ax2.plot_wireframe(X1_, X2_, mean_, color='black', lw=0.5, rcount=25, ccount=25)
    zlims = ax2.get_zlim()
    zmin, zmax = zlims[0] - 1, zlims[1]
    ax2.set_zlim(zmin, zmax)
    ax2.contour(X1_, X2_, mean_, cmap=cmap, alpha=alpha, linestyles="solid", zdir="z", offset=zmin)
    ax2.set_title('Surface Predicted Mean', fontsize=fs)
    ax2.set_xlabel(ax_labels[0], fontsize=fs, x=-0.1)
    ax2.set_ylabel(ax_labels[0], fontsize=fs, y=-0.1)
    ax2.tick_params(axis='both', labelsize=fs_ax-2)
    ax2.grid()
    fig.legend(loc=(0.025, 0.4), fontsize=fs)
    fig.suptitle(gen_title, fontsize=fs)
    cb = fig.colorbar(im2, ax=ax2, label=r'$\mu$')
    cb.ax.tick_params(labelsize=fs_ax) 
    if savepath:
        fig.tight_layout()
        fig.savefig(savepath)
    fig.show()