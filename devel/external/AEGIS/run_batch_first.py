import subprocess
import numpy as np
import glob
import os

def run_batch():
    for fl in glob.glob('outfile/out_1_*'):
        os.remove(fl)
    file_name ='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/seg_ids.txt'
    all_seg_ids = np.loadtxt(file_name, delimiter=" ",dtype='S2')
    for seg_id in all_seg_ids:
        print 'SEG ID ', seg_id
        outfile = 'outfile/out_1_{0}.txt'.format(seg_id)
        com1 = '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_full2/'
        com2 = 'python get_objects.py --out_path='+ com1
        final_args =['bsub', '-W' , '2:40', '-o', outfile, com2 ]
        final_args.append("--seg_id="+ seg_id)        
        subprocess.call(final_args)

if __name__ == '__main__':
    run_batch()