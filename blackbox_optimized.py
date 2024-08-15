from organizer import *


#set up random seed
org = Organizer()
org.set_random_seed()

#Set up the logger and the device
logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY,
                     '{}_{}_{}_User{}_C{}_Delay{}_{}_TrainEpoch{}_AttackEpoch{}_blackbox_op_{}_{}'.format(org.args.dataset, org.args.defense, org.args.attack,
                                                                                NUMBER_OF_PARTICIPANTS+NUMBER_OF_ADVERSARY, org.args.C, org.args.delay, org.args.dist, org.args.train_epoch, 
                                                                                org.args.max_epoch -  org.args.train_epoch, COVER_FACTOR, TIME_STAMP))

#strat the greybox attack
org.federated_training_black_box_optimized(logger)