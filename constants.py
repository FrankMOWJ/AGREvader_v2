from datetime import datetime
import torch

# General constants
PYTORCH_INIT = "PyTorch" #initialisation for pytorch`
now = datetime.now() #Current date and time
TIME_STAMP = now.strftime("%Y_%m_%d_%H_%M") #Time stamp for logging
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Device for training, you can use cpu or gpu

# delay 
T_DELAY = 0

# ratio of client to participate in training each round 
C = 1.0 # 0.5

# General hyper parameters for training
MAX_EPOCH = 200 #Maximum number of epochs # 1000
TRAIN_EPOCH = 0 #Number of epochs for training
BATCH_SIZE = 64 #Batch size for training
RESERVED_SAMPLE = 300 #Number of sample for covering attack # origin 300
INIT_MODE = PYTORCH_INIT # Initialisation mode
BATCH_TRAINING = True #Batch training or not

# Data set related
# You can explore more datasets here
LOCATION30 = "Location30" #Name of the dataset
LOCATION30_PATH = "./datasets-master/bangkok" #Path to the dataset
TRAIN_TEST_RATIO = (0.5, 0.5) #Ratio of training and testing data for Location30
LABEL_COL = 0 #Label column
LABEL_SIZE = 30 #Number of classes
DEFAULT_SET = LOCATION30 #Default dataset

COVID19 = "covid19"
COVID19_PATH = "./datasets-master/covid19"

CIFAR10 = "CIFAR10" 
CIFAR10_PATH = "~/.torch/"

CIFAR100 = "CIFAR100" 
CIFAR100_PATH = "~/.torch/"

MNIST = 'MNIST'
MNIST_PATH = "~/.torch/"

FASHION_MNIST = 'FASHION_MNIST'
FASHION_MNIST_PATH = '~/.torch/'

PURCHASE100 = "Purchase100"
PURCHASE100_PATH = "./datasets-master/purchase100.npz"

TEXAS100 = "Texas100"
TEXAS100_PATH = "./datasets-master/texas100.npz"

CINIC10 = 'CINIC10'
CINIC10_PATH = "./datasets-master/CINIC-10"

EMOTION = 'EMOTION10'
EMOTION_PATH = './datasets-master/Emotion_classification/'

SVHN = 'SVHN'
SVHN_PATH = '~/.torch/'

GTSRB = 'GTSRB'
GTSRB_PATH = './datasets-master/GTSRB'

SUN397 = 'SUN397'
SUN397_PATH = '~/.torch/'

STL10 = 'STL10'
STL10_PATH = './datasets-master/STL10'

FER2013 = 'FER2013'
FER2013_PATH = './datasets-master/fer2013'

DATASET = CIFAR10
# Attack method
ATTACK = 'None'

# Robust AGR
NONE = "None" #Average
MEDIAN = "Median" #Robust AGR -Median
FANG = "Fang" #Robust AGR -Fang

TRIM = "Trim" #Robust AGR -Trim
TRIM_BOUND = 1 #Trim bound

KRUM = "Krum" #Robust AGR -Krum

MULTI_KRUM = "Multi-Krum" #Robust AGR -Multi-Krum
MULTI_KRUM_K = 4 #Number of Krum

DEEPSIGHT = "Deepsight" #Robust AGR -DeepSight
NUM_CLUSTER = 2

RFLBAT = "Rflbat"
NUM_COMPONENTS = 2

FLAME = "Flame"

FOOLSGOLD = "Foolsgold"

ANGLE_MEDIAN = 'Angle-Median'

ANGLE_TRIM = 'Angle-Trim'
ANGLE_TRIM_BOUND = 2

TOPK = "Topk"

DIFFERENTIAL_PRIVACY = "Differential-Privacy"

# Federated learning parameters
NUMBER_OF_PARTICIPANTS = 9 #Number of participants
PARAMETER_EXCHANGE_RATE = 1 #Parameter exchange rate
PARAMETER_SAMPLE_THRESHOLD = 1 #Parameter sample threshold
GRADIENT_EXCHANGE_RATE = 1 #Gradient exchange rate
GRADIENT_SAMPLE_THRESHOLD = 1 #Gradient sample threshold

# Attacker related
NUMBER_OF_ADVERSARY = 1 #Number of adversaries
NUMBER_OF_ATTACK_SAMPLES = 300 #Number of attack samples
ASCENT_FACTOR = 1 #Ascent factor for Gradient Ascent
BLACK_BOX_MEMBER_RATE = 0.5 # Member sample rate for black box attack
FRACTION_OF_ASCENDING_SAMPLES = 1 #Fraction of ascending samples
COVER_FACTOR = 0.5 #Cover factor for covering attack
GREY_BOX_SHUFFLE_COPIES = 5# Attacker shuffle parameter for related for Greybox attack
MISLEAD_FACTOR = 0.8 # Mislead factor for Mislead part


# IO related
EXPERIMENTAL_DATA_DIRECTORY = "./CIFAR10_output_5/" # ./CIFAR10_output_syn/"


# Random seed
GLOBAL_SEED = 999


