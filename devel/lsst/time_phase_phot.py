import os, sys
import cProfile, pstats

def time_geom():
    # Mock a command-line arguments object so we can run in the current process
    class Args(object):
        seed = 12345
        n = 100000
        jmax = 22
        ell = 4.0
        sigma = 0.05
        r0_500 = 0.2
        nlayers = 6
        time_step = 10
        screen_size = 409.6
        screen_scale = 0.1
        max_speed = 20.0
        lam = 700.0
        diam = 8.36
        obscuration = 0.61
        nstruts = 0
        strut_thick = 0.05
        strut_angle = 0.0
        nx = 64
        size = 0.2 * 64  # -> pixel_scale=0.2
        accumulate = False
        pad_factor = 1.0
        oversampling = 1.0
        geom_oversampling = 1.0
        geom_nphot = 10
        vmax = 1.e-3
        out = "output/time_geom_"
        do_fft=0
        do_geom=1
        make_movie=0
        make_plots=0
    original_dir = os.getcwd()
    try:
        os.chdir('../../examples')
        new_dir = os.getcwd()
        if new_dir not in sys.path:
            sys.path.append(new_dir)
        import fft_vs_geom_movie
        pr = cProfile.Profile()
        pr.enable()

        fft_vs_geom_movie.make_movie(Args)

        pr.disable()
        ps = pstats.Stats(pr).sort_stats('cumtime')
        ps.print_stats(30)
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.print_stats(30)
    finally:
        os.chdir(original_dir)


time_geom()
