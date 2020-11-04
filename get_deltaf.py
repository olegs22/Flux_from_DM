import numpy as np
import argparse
import h5py
from nbodykit.lab import *
from Lya_tools import E,tau
from nbodykit.utils import GatherArray, ScatterArray

parser = argparse.ArgumentParser()
parser.add_argument('--idir',type=str)
parser.add_argument('--filename',type=str)
parser.add_argument('--output',type=str)
parser.add_argument('--box',type=float,default=1000.)
parser.add_argument('--nmesh',type=int,default=1024)
parser.add_argument('--z',type=float,default=2.5)
args = parser.parse_args()

comm = CurrentMPIComm.get()
rank = comm.rank

def getflux(v):
    op = tau(v,args.z)
    F = np.exp(-1.0*op)
    return F

def save_deltaf(output,deltaf):
    File_h5py = output+'.hdf5'
    f = h5py.File(File_h5py,'w')
    grid = f.create_group("grids")
    ds = grid.create_dataset("delta",data=deltaf)
    f.close()
    return None

def read_deltaf(file,boxsize):
    f = h5py.File(file,'r')
    var = f["/grids/delta"]
    return ArrayMesh(var,BoxSize=boxsize,comm=comm,root=0)

def get_flux_field(file_name,output,z,Nside,boxsize):

    #tick_to_mesh = time.time()
    cat = Gadget1Catalog(file_name,
    columndefs=[('Position', ('auto', 3), 'all'),
    ('GadgetVelocity', ('auto', 3), 'all'),('Mass', 'auto', None)],ptype=1)


    H = 100.0 * E(omega_m=0.308, omega_de=0.692, redshift=z)
    rsd_factor = np.sqrt(1.0+z) / H

    #this next two lines only add rsd to a give LoS. In this case is the z direction 
    los = [0,0,1]
    cat['Position'] = cat['Position'] + rsd_factor * cat['GadgetVelocity'] * los

    mesh = cat.to_mesh(resampler='cic', BoxSize=boxsize,Nmesh=Nside, compensated=True)
    mesh_array = mesh.preview().flatten()
    mesh_scatter = ScatterArray(mesh_array,comm,root=0)
    flux = getflux(mesh_scatter)
    
    #this is a cubic mesh with the flux value 
    flux_complete = GatherArray(flux,comm,root=0) 
    #this save the mesh as an array in a hdf5 file
    save_deltaf(output,flux_complete) 

    # if you want to calculate the flux deltas use the next snippet
    """
    if rank == 0:
        mean_f = np.mean(np.asarray(flux_complete),axis=0)
        delta_f = (np.asarray(flux_complete)/mean_f) - 1.0
        delta_f = delta_f.reshape(Nside,Nside,Nside)
        save_deltaf(outfile,delta_f)
    else :
        delta_f=None
    """

    #this mesh can be directly used if ProjectedFFTpower
    #if you want the deltaf mesh change flux_complete to delta_f
    return ArrayMesh(flux_complete,BoxSize=boxsize,comm=comm,root=0) 
   
if __name__ == "__main__":
    import glob as glob

    dump = glob.glob(args.idir+'/*')
    for i in range(len(dump)):
        file = dump[i] +'/'+ args.filename + '.*'
        outfile = args.output + '_' + str(i)
        get_flux_field(file,outfile,args.z,args.nmesh,args.box)