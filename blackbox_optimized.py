from organizer import *


#Set up the logger and the device
logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY,
                     '{}_{}_{}_User{}_TrainEpoch{}_AttackEpoch{}_blackbox_op_{}_{}'.format(DATASET, DEFAULT_AGR, ATTACK, 
                                                                                NUMBER_OF_PARTICIPANTS+NUMBER_OF_ADVERSARY, TRAIN_EPOCH, MAX_EPOCH - TRAIN_EPOCH, 
                                                                                COVER_FACTOR, TIME_STAMP))
#set up random seed
org = Organizer()
org.set_random_seed()
#strat the greybox attack
org.federated_training_black_box_optimized(logger)