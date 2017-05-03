This is a python implementation of the Single Matching Phase (SMP) stereo matching algorithm described in the paper:

  Di Stefano, Luigi, Massimiliano Marchionni, and Stefano Mattoccia. "A fast area-based stereo matching algorithm." Image and vision computing 22.12 (2004): 983-1005.

To use it, you will need to download some image data from http://vision.middlebury.edu/stereo/data/. I used the 2001 datasets, you just need the first two actual images of each picture (not the ground truth depth maps, that's what the program is for). Make sure to put them in the empty data/ directory, with the following structure:

data/
    map/
        im0.pgm
        im1.pgm
    venus/
        im0.ppm
        im1.ppm

Program can be run from the command line by:

  python stereo_matching.py 'picture_to_run_on'

Where 'picture_to_run_on' is not the name of a particular file but rather the name of one of the given pictures, aka: map, tsukuba, sawtooth, poster, venus, bull, barn1, barn2

Example:
  python stereo_matching.py map

**Note:** This program can be run on any picture, but I've hardcoded these 8 to run with command-line arguments. If you want to run others you'll have to open the .py file, scroll down to the main method, and put in your own path/filename where all the images are loaded.
