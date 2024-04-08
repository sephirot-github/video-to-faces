import argparse
from .main import video_to_faces

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_path', metavar='', help='Path to a video file, a directory with video files, or a .txt file with full paths inside.')
parser.add_argument('-e', '--input_ext', metavar='', help='If -i is a directory, process only files with this extension. Can contain multiple extensions separated by semicolon.')
parser.add_argument('-o', '--out_dir', metavar='', help='Path for the output. If omitted, the results will be saved alongside the input.')
parser.add_argument('-op', '--out-prefix', metavar='', help='A prefix to add to the automatic name "[k][n].jpg" ...')
parser.add_argument('-s', '--style', metavar='', required=True, choices=['live', 'anime'], help='Whether the inputs are anime or live-action videos. Necessary for choosing the proper models.')
parser.add_argument('-m', '--mode', metavar='', choices=['full', 'detection', 'grouping'], help='Allows to perform detection and grouping steps separately (default is full). For grouping, specify either -o or the same -i used during detection.')
parser.add_argument('-d', '--device', metavar='', choices=['cpu', 'cuda', 'cuda:0'], help='Whether to use CPU or GPU for processing. Defaults to GPU if one is present.')

args = parser.parse_args()
video_to_faces(**vars(args))