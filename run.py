# -*- coding: utf-8 -*-
"""
Main file to run app on AWS.

Author: Stephane Damolini
Site: LooksLikeWho.damolini.com 
"""

# IMPORTS
import os
import tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("***TF-GPU DISABLED***")
from app import app
import argparse
import gc


# START THE APP
if __name__ == '__main__':

    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())
    gc.collect()

    if args["debug"]:
        print("***DEBUG***")
        app.run(host='0.0.0.0', threaded=True, port=80, debug=True)
    else:
        app.run(host='0.0.0.0', threaded=True, port=80)
        
# END OF CODE