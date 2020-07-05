from mpi4py import MPI

from run_master import *
from run_worker import *

import argparse

parser = argparse.ArgumentParser(description='Running Nevergrad on multiple CPUs using MPI')
parser.add_argument('--opt', help='optimizer to use', required=True )
parser.add_argument('--budget', help='budget for optimizer', required=True, type=int )
parser.add_argument('--hours', help='How long to run the simulation (budget just needs to be an estimate). This time does not include spin down time.', required=False, type=float, default=-1.0 )
parser.add_argument('--out_prefix', help='prefix for output files', required=True, type=str )
parser.add_argument('--in_prefices', help='comma separated list of prefices to load from', required=False, type=str, default="" )
args = parser.parse_args()

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

print( args.opt, rank )

if rank == 0:
    run_master( comm=comm, nprocs=nprocs, rank=rank, opt=args.opt, budget=args.budget, out_prefix=args.out_prefix, in_prefices=args.in_prefices, hours=args.hours )
else:
    run_worker( comm, rank, out_prefix=args.out_prefix )
