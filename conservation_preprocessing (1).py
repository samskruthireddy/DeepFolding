
import numpy as np
from prody import parseMSA
from glob2 import glob
import os

dict = {}

current_path = os.getcwd( )
os.chdir( 'datadirectory' )

count = 0

for pfam_name in glob( "*_full.fasta" ):

    count = count + 1

    print "\nParsing file number %d , name %s \t" %( count , pfam_name )

    if os.stat( pfam_name ).st_size == 0:
        continue

    try:
        MSA = parseMSA( pfam_name )
    except Exception:
        print( "Could not parse MSA" )
        continue

    S = MSA.getArray( )

    dict[ pfam_name ] = np.asarray( [ np.sum( np.logical_or( S == chr( c + 65 ) , S == chr( c + 97 ) ) , axis = 0 ) for c in range( 26 ) ] ) / float( S.shape[ 0 ] )

np.save( 'PSSM.npy' , dict )
os.chdir( current_path )
~                                      