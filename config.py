
import os
args = dict()


#project structure
#args["CODE_DIRECTORY"] = os.path.abspath(".")   #absolute path to the code directory
#args["DATA_DIRECTORY"] = None   #absolute path to the data directory
#args["DEMO_DIRECTORY"] = args["DEMO_DIRECTORY"] = r"C:/Users/lenovo/Desktop/OneDrive/Pictures/Camera Roll"   #absolute path to the demo directory
#args["PRETRAINED_MODEL_FILE"] = os.path.abspath("./final/models/pretrained_model.pt")    #relative path to the pretrained model file
#args["TRAINED_MODEL_FILE"] = "/final/models/trained_model.pt"   #relative path to the trained model file
#args["TRAINED_LM_FILE"] = None  #absolute path to the trained language model file
#args["TRAINED_FRONTEND_FILE"] = None #absolute path to the trained visual frontend file

args["CODE_DIRECTORY"] = os.path.abspath(".")  # Absolute path to the code directory
args["DATA_DIRECTORY"] = None  # Not needed since we're running a demo

# Absolute path to the demo directory
args["DEMO_DIRECTORY"] = r"C:/Users/lenovo\Desktop/OneDrive/Pictures/Camera Roll"

# Absolute path to the pretrained model file
args["PRETRAINED_MODEL_FILE"] = os.path.abspath("./final/models/pretrained_model.pt")

# Absolute path to the trained model file (for inference)
args["TRAINED_MODEL_FILE"] = os.path.join(args["CODE_DIRECTORY"], "final/models/video-only.pt")


# Ignore these since we're not training
args["TRAINED_LM_FILE"] = os.path.abspath("final/models/language_model.pt")
args["TRAINED_FRONTEND_FILE"] = os.path.abspath("final/models/visual_frontend.pt")




#data
args["PRETRAIN_VAL_SPLIT"] = 0.01   #validation set size fraction during pretraining
args["NUM_WORKERS"] = 8 #dataloader num_workers argument
args["PRETRAIN_NUM_WORDS"] = 1  #number of words limit in current curriculum learning iteration
args["MAIN_REQ_INPUT_LENGTH"] = 145 #minimum input length while training
args["CHAR_TO_INDEX"] = {" ":1, "'":22, "1":30, "0":29, "3":37, "2":32, "5":34, "4":38, "7":36, "6":35, "9":31, "8":33,
                         "A":5, "C":17, "B":20, "E":2, "D":12, "G":16, "F":19, "I":6, "H":9, "K":24, "J":25, "M":18,
                         "L":11, "O":4, "N":7, "Q":27, "P":21, "S":8, "R":10, "U":13, "T":3, "W":15, "V":23, "Y":14,
                         "X":26, "Z":28, "<EOS>":39}    #character to index mapping
args["INDEX_TO_CHAR"] = {1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5", 38:"4", 36:"7", 35:"6", 31:"9", 33:"8",
                         5:"A", 17:"C", 20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H", 24:"K", 25:"J", 18:"M",
                         11:"L", 4:"O", 7:"N", 27:"Q", 21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V", 14:"Y",
                         26:"X", 28:"Z", 39:"<EOS>"}    #index to character reverse mapping


#preprocessing
args["VIDEO_FPS"] = 25  #frame rate of the video clips
args["ROI_SIZE"] = 112 #height and width of input greyscale lip region patch
args["NORMALIZATION_MEAN"] = 0.4161 #mean value for normalization of greyscale lip region patch
args["NORMALIZATION_STD"] = 0.1688  #standard deviation value for normalization of greyscale lip region patch


#training
args["SEED"] = 19220297 #seed for random number generators
args["BATCH_SIZE"] = 32 #minibatch size
args["STEP_SIZE"] = 16384   #number of samples in one step (virtual epoch)
args["NUM_STEPS"] = 1000 #maximum number of steps to train for (early stopping is used)
args["SAVE_FREQUENCY"] = 10 #saving the model weights and loss/metric plots after every these many steps


#optimizer and scheduler
args["INIT_LR"] = 1e-4  #initial learning rate for scheduler
args["FINAL_LR"] = 1e-6 #final learning rate for scheduler
args["LR_SCHEDULER_FACTOR"] = 0.5   #learning rate decrease factor for scheduler
args["LR_SCHEDULER_WAIT"] = 25  #number of steps to wait to lower learning rate
args["LR_SCHEDULER_THRESH"] = 0.001 #threshold to check plateau-ing of wer
args["MOMENTUM1"] = 0.9 #optimizer momentum 1 value
args["MOMENTUM2"] = 0.999   #optimizer momentum 2 value


#model
args["NUM_CLASSES"] = 40    #number of output characters


#transformer architecture
args["PE_MAX_LENGTH"] = 2500    #length up to which we calculate positional encodings
args["TX_NUM_FEATURES"] = 512   #transformer input feature size
args["TX_ATTENTION_HEADS"] = 8  #number of attention heads in multihead attention layer
args["TX_NUM_LAYERS"] = 6   #number of Transformer Encoder blocks in the stack
args["TX_FEEDFORWARD_DIM"] = 2048   #hidden layer size in feedforward network of transformer
args["TX_DROPOUT"] = 0.1    #dropout probability in the transformer


#beam search
#args["BEAM_WIDTH"] = 100    #beam width
#args["LM_WEIGHT_ALPHA"] = 0.5   #weight of language model probability in shallow fusion beam scoring
#args["LENGTH_PENALTY_BETA"] = 0.1   #length penalty exponent hyperparameter
#args["THRESH_PROBABILITY"] = 0.0001 #threshold probability in beam search algorithm

#001 Prediction: WHAT IS I THINK I THINK WORK
# args["BEAM_WIDTH"] = 20
# args["LM_WEIGHT_ALPHA"] = 0.7
# args["LENGTH_PENALTY_BETA"] = 0.3
# args["THRESH_PROBABILITY"] = 0.05
# args["USE_LM"] = True  #whether to use language model for decoding

args["BEAM_WIDTH"] = 20
args["LM_WEIGHT_ALPHA"] = 0.7
args["LENGTH_PENALTY_BETA"] = 0.3
args["THRESH_PROBABILITY"] = 0.05
args["USE_LM"] = True  #whether to use language model for decoding


#testing
args["TEST_DEMO_DECODING"] = "search"   #test/demo decoding type - "greedy" or "search"


if __name__ == "__main__":

    for key,value in args.items():
        print(str(key) + " : " + str(value))