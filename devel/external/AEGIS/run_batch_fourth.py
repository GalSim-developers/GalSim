import subprocess
import glob
import os
import numpy as np

def main(args):
    for fl in glob.glob('outfile/out_4_*'):
        os.remove(fl)
    all_seg = np.loadtxt(args.seg_file_name, delimiter=" ",dtype='S2')
    print "path is", args.main_path
    for f, filt in enumerate(args.filter_names):
        for seg_id in all_seg:
            print 'SEG ID: ', seg_id, ' filter: ', filt 
            outfile = 'outfile/out_4_{0}.txt'.format(seg_id)
            com1 = args.main_path
            com = 'python get_cat_seg.py --seg_id='+ seg_id+ ' --main_path='+ com1
            final_args =['bsub', '-W' , '0:35','-o', outfile, com]
            final_args.append("--filter="+ filt)
            final_args.append("--file_filter_name="+ args.filter_file_names[f])    
            subprocess.call(final_args)
              
if __name__ == '__main__':
    import subprocess
    import galsim
    import numpy as np
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--filter_names', default= ['f606w','f814w'],
                        help="names of filters [Default: ['f606w','f814w']]")
    parser.add_argument('--filter_file_names', default= ['V','I'],
                        help="Names of filters to write inf file [Default: ['V','I']]")
    parser.add_argument('--main_path',
                        default = '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_full/')
    parser.add_argument('--seg_file_name', default ='/nfs/slac/g/ki/ki19/deuce/AEGIS/unzip/seg_ids.txt',
                        help="file with all seg id names" )
    args = parser.parse_args()
    main(args)




