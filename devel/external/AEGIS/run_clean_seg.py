import subprocess
import numpy as np
import glob
import os
from astropy.table import Table



def run_clean_seg(args):
    seg_id= args.seg_id
    for fl in glob.glob('outfile/out_3_' + seg_id +'_*'):
        os.remove(fl)
    print 'SEG ID ', seg_id
    path = '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_full/'
    obj_file =  path + seg_id + '/objects_with_p_stamps.txt'
    obj_list = np.loadtxt(obj_file, dtype=int)   
    for num in  obj_list:
        outfile = 'outfile/out_3_{0}_{1}.txt'.format(seg_id, num)
        com = 'python clean_pstamp.py --main_path='+ args.main_path
        final_args =['bsub','-W', '0:35', com]
        final_args.append("--seg_id=" + seg_id)
        final_args.append("--num=" + str(num))
        subprocess.call(final_args)

if __name__ == '__main__':
    import subprocess
    import numpy as np
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--seg_id', default= '0a',
                        help="id of segment to run [Default:'0a']")
    parser.add_argument('--main_path', 
                        default= '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_full/',
                        help="Path where image files are stored \
                        [Default:'/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_full/'] ")
    args = parser.parse_args()
    run_clean_seg(args)

