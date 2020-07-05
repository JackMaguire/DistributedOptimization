from mpi4py import MPI
import nevergrad as ng
import numpy as np
import time

all_results_dofs = []
all_results_scores = []

def send_job_to_node( comm, dofs, node, tag=1 ):
    comm.send( dofs, dest=node, tag=tag )

def interpret_result( bundle ):
    dofs = bundle[ 0 ]
    score = bundle[ 1 ]
    print( "RESULT", score, dofs )

    all_results_dofs.append( np.asarray( dofs.value ) )
    all_results_scores.append( np.asarray( score ) )

def tell_node_to_die( comm, node ):
    send_job_to_node( comm, "die", node, tag=0 )
    message = comm.recv( source=node, tag=MPI.ANY_TAG )
    if message != 0:
        print( "Node ", node, " sent ", message, " instead of 0 upon kill" )
    
def execute_kill_seq( comm, available_nodes, working_nodes ):

    while len( available_nodes ) > 0:
        node = available_nodes.pop()
        tell_node_to_die( comm, node )
        
    while len( working_nodes ) > 0:
        status = MPI.Status()
        bundle = comm.recv( source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status )
        interpret_result( bundle )
        
        source = status.Get_source()
        working_nodes.remove( source )
        tell_node_to_die( comm, source )

def keep_going( hours_elapsed, hours_limit, njobs_sent, budget ):
    if hours_limit > 0:
        return hours_elapsed < hours_limit
    else:
        return njobs_sent < budget
        
#https://stackoverflow.com/questions/21088420/mpi4py-send-recv-with-tag
def run_master( comm, nprocs, rank, opt, budget, out_prefix, in_prefices, hours ):
    
    available_nodes = set()
    for i in range( 1, nprocs ):
        available_nodes.add( i )

    working_nodes = set()

    #TODO
    Params = ng.p.Array( shape=(3,) )
    optimizer = ng.optimizers.registry[ opt ]( parametrization=Params, budget=budget, num_workers=(nprocs-1) )

    #Load if needed
    if len( in_prefices ) > 0:
        npoints_loaded=0
        for prefix in in_prefices.split( "," ):
            filenamed = prefix + ".all_results_dofs.npy"
            filenames = prefix + ".all_results_scores.npy"
            dofs = np.load( filenamed, allow_pickle=False )
            score = np.load( filenames, allow_pickle=False )
            assert( len( dofs ) == len( score ) )
            for i in range( 0, len( dofs ) ):
                #optimizer.suggest( dofs[ i ] )
                #x = optimizer.ask()
                x = optimizer.parametrization.spawn_child(new_value=dofs[i] )
                optimizer.tell( x, score[ i ] )
                npoints_loaded += 1
        print( "loaded", npoints_loaded, "points" )
        
    
    njobs_sent = 0
    begin = time.time()
    while keep_going( hours_elapsed=float(time.time()-begin)/3600.0, hours_limit=hours, njobs_sent=njobs_sent, budget=budget ):
    #for b in range( 0, budget ):
        if njobs_sent % 1 == 0:
            print( "Sent", njobs_sent, "jobs from budget of", budget )
        if len( available_nodes ) == 0:
            #All are busy, wait for results
            status = MPI.Status()
            bundle = comm.recv( source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status )
            source = status.Get_source()
            working_nodes.remove( source )
            available_nodes.add( source )

            dofs = bundle[ 0 ]
            score = bundle[ 1 ]
            optimizer.tell( dofs, score )

            interpret_result( bundle )
        #end if
        
        dofs = optimizer.ask()

        node = available_nodes.pop() #removes node from available_nodes
        send_job_to_node( comm, dofs, node )
        working_nodes.add( node )
        njobs_sent += 1

    print( "Spinning down after", time.time() - begin, "seconds"  )
    #end for b    
    execute_kill_seq( comm, available_nodes, working_nodes )
    print( "Finished after", time.time() - begin, "seconds"  )
    print( "Ran", njobs_sent, "jobs" )
    print( optimizer.provide_recommendation().value )

    test1 = np.asarray( all_results_dofs )
    print( test1.shape )
    
    np.save( out_prefix + ".all_results_dofs.npy", np.asarray(all_results_dofs), allow_pickle=False )
    np.save( out_prefix + ".all_results_scores.npy", np.asarray(all_results_scores), allow_pickle=False )
