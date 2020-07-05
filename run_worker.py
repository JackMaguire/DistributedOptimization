from mpi4py import MPI
from score_dofs import *

def score( dofs ):
    #USER TODO
    return {"score":0}

def run_worker( comm, rank, out_prefix ):

    dumped_pose_count = 0;
    
    while True:
        status = MPI.Status()
        dofs = comm.recv( source=0, tag=MPI.ANY_TAG, status=status )
        if status.Get_tag() == 0:
            comm.send( 0, dest=0, tag=0 )
            break

        score_dict = score_dofs( dofs )
        
        assert( "score" in return_dict )
        final_score = score_dict[ "score" ]

        bundle = [ dofs, final_score ]
        comm.send( bundle, dest=0, tag=1 )
