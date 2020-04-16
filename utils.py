import argparse

ATOL = 1e-5
RTOL = 1e-5

def argparse_inference():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nbatch', default=1, type=int,
                        help="number of batches to be tested")
    parser.add_argument("--verbosity", default=False, action='store_true',
                        help="increase output verbosity")
    parser.add_argument("--print-internal", default=False, action='store_true',
                        help="print internal buffers")
    parser.add_argument("--dry", default=False, action='store_true',
                        help="dry run")
    parser.add_argument('--filename', default="trained_model", type=str,
                        help="filename of the saved model")
    return parser.parse_args()
