from organizer import *


#set up random seed
org = Organizer()
org.set_random_seed()

#Set up the logger and the device
logger = make_logger("project", org.args.output_dir,
                     '{}_{}_{}_Client{}_Malicious{}_C{}_Delay{}_{}_AttackSample{}_NoniidBias{}_TrainEpoch{}_AttackEpoch{}_blackbox_op_{}_{}'.format(org.args.dataset, org.args.defense, org.args.attack,
                                                                                org.args.Number_client, org.args.Number_malicious, org.args.C, org.args.delay, org.args.dist, 
                                                                                org.args.number_attack_samples, org.args.bias, org.args.train_epoch, org.args.max_epoch -  org.args.train_epoch, COVER_FACTOR, TIME_STAMP))

#strat the greybox attack
org.federated_training_black_box_optimized(logger)