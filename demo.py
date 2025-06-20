"""

Note -
Video length should be such that the expected transcription length is less than 100 characters.
For this, a long video can be appropriately segmented into clips of appropriate length
depending on the speaking rate of the speaker. For a speaker with around 160 words per min,
and considering 6 characters per word (including space) on average, clip lengths should be
around 6 secs.
The predicted transcriptions will have all characters in capital with no punctuations
other than an apostrophe (').
"""

import torch
import numpy as np
import cv2 as cv
import os

from config import args
from models.video_net import VideoNet
from models.visual_frontend import VisualFrontend
from models.lrs2_char_lm import LRS2CharLM
from data.utils import prepare_main_input, collate_fn
from utils.preprocessing import preprocess_sample, split_video
from utils.decoders import ctc_greedy_decode, ctc_search_decode

def main():
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")

    if args["TRAINED_MODEL_FILE"] is not None:
        print("\nTrained Model File: %s" %(args["TRAINED_MODEL_FILE"]))
        print("\nDemo Directory: %s" %(args["DEMO_DIRECTORY"]))

        # Declaring the model and loading the trained weights
        model = VideoNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
                         args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
        model.load_state_dict(torch.load(args["TRAINED_MODEL_FILE"], map_location=device))
        model.to(device)

        # Declaring the visual frontend module
        vf = VisualFrontend()
        vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"], map_location=device))
        vf.to(device)

        # Declaring the language model
        lm = LRS2CharLM()
        lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"], map_location=device))
        lm.to(device)
        if not args["USE_LM"]:
            lm = None

        print("\n\nRunning Demo .... \n")

        # Walking through the demo directory and running the model on all video files in it
        for root, dirs, files in os.walk(args["DEMO_DIRECTORY"]):
            for file in files:
                if file.endswith(".mp4"):
                    sampleFile = os.path.join(root, file[:-4])

                    # Split longer videos into 6-second chunks
                    clip_files = split_video(sampleFile + ".mp4", args["VIDEO_FPS"], clip_duration=6)

                    full_prediction = ""

                    for clip_file in clip_files:
                        # Preprocessing the sample
                        params = {
                            "roiSize": args["ROI_SIZE"],
                            "normMean": args["NORMALIZATION_MEAN"],
                            "normStd": args["NORMALIZATION_STD"],
                            "vf": vf
                        }
                        preprocess_sample(clip_file[:-4], params)

                        # Converting the data sample into appropriate tensors for input to the model
                        visualFeaturesFile = clip_file[:-4] + ".npy"
                        videoParams = {"videoFPS": args["VIDEO_FPS"]}
                        inp, _, inpLen, _ = prepare_main_input(visualFeaturesFile, None, args["MAIN_REQ_INPUT_LENGTH"],
                                                               args["CHAR_TO_INDEX"], videoParams)
                        inputBatch, _, inputLenBatch, _ = collate_fn([(inp, None, inpLen, None)])

                        # Running the model
                        inputBatch = inputBatch.float().to(device)
                        inputLenBatch = inputLenBatch.int().to(device)
                        model.eval()
                        with torch.no_grad():
                            outputBatch = model(inputBatch)

                        # Obtaining the prediction using CTC decoder
                        if args["TEST_DEMO_DECODING"] == "greedy":
                            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch,
                                                                                   args["CHAR_TO_INDEX"]["<EOS>"])

                        elif args["TEST_DEMO_DECODING"] == "search":
                            beamSearchParams = {
                                "beamWidth": args["BEAM_WIDTH"],
                                "alpha": args["LM_WEIGHT_ALPHA"],
                                "beta": args["LENGTH_PENALTY_BETA"],
                                "threshProb": args["THRESH_PROBABILITY"]
                            }
                            predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch,
                                                                                    beamSearchParams,
                                                                                    args["CHAR_TO_INDEX"][" "],
                                                                                    args["CHAR_TO_INDEX"]["<EOS>"], lm)
                        else:
                            print("Invalid Decode Scheme")
                            exit()

                        # Converting character indices back to characters
                        pred = predictionBatch[:][:-1]
                        pred = "".join([args["INDEX_TO_CHAR"][ix] for ix in pred.tolist()])

                        # Append to final prediction
                        full_prediction += pred + " "

                    # Printing the predictions for the entire video
                    print("File: %s" % (file))
                    print("Final Prediction: %s" % (full_prediction.strip()))
                    print("\n")

        print("Demo Completed.\n")

    else:
        print("\nPath to trained model file not specified.\n")

    return

if __name__ == "__main__":
    main()
