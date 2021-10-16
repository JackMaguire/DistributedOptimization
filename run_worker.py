from mpi4py import MPI

import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta import core, protocols
from pyrosetta.rosetta.protocols.peptide import BindToCO, BindToNH

binder = BindToNH()

def scale_param( param, minval, maxval ):
    f = (param + 2.5) / 5.0
    return minval + ((maxval-minval)*f)

class Tracker:
    def __init__( self, dofs ):
        self.dofs = dofs
        self.d_count = 0

    def get( self ):
        #print( self.dofs )
        assert( self.d_count < 10 )
        x = self.dofs.value[ self.d_count ]
        #print( "X", x )
        self.d_count += 1
        return x
        

def score_dofs( dofs, pose, NHresid, sfxn ):

    global binder

    x = Tracker( dofs )

    newpose = pose_from_sequence( "GGG" )
    newpose.pdb_info( core.pose.PDBInfo( newpose, True ) )
    newpose.pdb_info().add_reslabel( 1, "__PEPTIDE__" )
    newpose.pdb_info().add_reslabel( 2, "__PEPTIDE__" )
    newpose.pdb_info().add_reslabel( 3, "__PEPTIDE__" )

    newpose.set_psi( 1, scale_param( x.get(), -225.0, 225.0 ) )
    newpose.set_phi( 2, scale_param( x.get(), -225.0, 225.0 ) )
    newpose.set_psi( 2, scale_param( x.get(), -225.0, 225.0 ) )
    newpose.set_phi( 3, scale_param( x.get(), -225.0, 225.0 ) )

    CA_N_H_O_torsion_angle_rad = scale_param( x.get(), -4, 4 )
    N_H_O_bond_angle_rad = scale_param( x.get(), 0.001, 3.14 )
    H_O_dist_A = scale_param( x.get(), 1, 3 )

    N_H_O_C_torsion_angle_rad = scale_param( x.get(), -4, 4 )
    H_O_C_bond_angle_rad = scale_param( x.get(), 0.001, 3.14 )
    H_O_C_CA_torsion_angle_rad = scale_param( x.get(), -4, 4 )

    #print( "CA_N_H_O_torsion_angle_rad", CA_N_H_O_torsion_angle_rad )

    binder.run(
        pose, NHresid,
        newpose, 2,

        CA_N_H_O_torsion_angle_rad,
        N_H_O_bond_angle_rad,
        H_O_dist_A,

        N_H_O_C_torsion_angle_rad,
        H_O_C_bond_angle_rad,
        H_O_C_CA_torsion_angle_rad
    )

    score = sfxn.score( pose )

    return {"score":score}

def run_worker( comm, rank, out_prefix, posename, NHres ):

    pyrosetta.init( "-mute all" )
    sfxn = get_score_function()

    baseline_pose = pose_from_file( posename )

    selector = core.select.residue_selector.ResidueIndexSelector( NHres )
    selection = selector.apply( baseline_pose )
    selected = [ i for i in range( 1, len(selection)+1 ) if selection[i] ]
    print( selected )
    assert( len(selected) == 1 )
    resid = selected[0]

    dumped_pose_count = 0
    
    best_pose = Pose()
    best_score = sfxn.score( baseline_pose ) + 10

    while True:
        status = MPI.Status()
        dofs = comm.recv( source=0, tag=MPI.ANY_TAG, status=status )
        if status.Get_tag() == 0:
            comm.send( 0, dest=0, tag=0 )
            break

        pose = Pose()
        pose.assign( baseline_pose )
        score_dict = score_dofs( dofs, pose, resid, sfxn )
        
        assert( "score" in score_dict )
        final_score = score_dict[ "score" ]

        if final_score < best_score:
            print( "NEW BEST POSE" )
            best_score = final_score
            best_pose.assign( pose )
            best_pose.dump_pdb( "best_pose.pdb" )

        bundle = [ dofs, final_score ]
        comm.send( bundle, dest=0, tag=1 )
