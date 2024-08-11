from models import *
from constants import *
import pandas as pd
import numpy as np
import os, random
from data_reader import DataReader
import argparse


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    
    parser.add_argument(
        "-a",
        "--attack",
        help="attacker attack type",
        choices=['norm', 'unit', 'angle'],
        default='angle'
    )

    parser.add_argument(
        "-d",
        "--defense",
        help="normal users defense type",
        choices=['Fang', 'Median', 'Trim', 'None'],
        default='median'
    )
    
    parser.add_argument(
        "-dt",
        "--dataset",
        help="dataset",
        required=True,
        choices=['CIFAR10', 'Location30', 'Purchase100'],
        default='CIFAR10'
    )

    parser.add_argument(
        "-C",
        help="ratio of clients to participate",
        required=True,
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--delay",
        help="max delay",
        required=True,
        type=int,
        default=0
    )

    # parser.add_argument(
    #     "-r",
    #     "--victim_ratio",
    #     help="customized victim parameter weight when combine victim and cover params, \
    #         used different ratio from the config file",
    #     default=None,
    #     type=float
    # )
    
    parser.add_argument(
        "--device",
        help="device index",
        default=0
    )
    
    parser.add_argument(
        "--dist",
        help="iid or non-iid setting",
        choices=['iid', 'non-iid'],
        default='iid',
        type=str
    )
    
    parser.add_argument(
        "--cover_times",
        help="Number of times to try cover set",
        default=None
    )
    
    # parser.add_argument(
    #     "-o",
    #     "--output_dir",
    #     help="log output direction",
    #     default='./results-agrevader',
    #     type=str
    # )

    args = parser.parse_args()
    return args
        
class Organizer():
    '''
    This class is used to organize the whole process of federated learning.
    '''

    def __init__(self):
        '''
        Initialize the organizer with random seed, data reader, target model.
        :param seed: random seed
        :param reader: data reader
        :param target: target model
        :param bar_recorder: bar recorder
        '''
        # parser
        self.args = get_parser()
        self.DATA_DISTRIBUTION = self.args.dist
        self.set_random_seed()
        self.reader = DataReader(data_set=DATASET, data_distribution=self.DATA_DISTRIBUTION)
        self.target = TargetModel(self.reader, participant_index=0, model=DATASET)

    def set_random_seed(self, seed=GLOBAL_SEED):
        '''
        Set random seed for reproducibility.
        :param seed: random seed defined in constants.py
        '''
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def federated_training_grey_box_optimized(self, logger, record_process=True,
                                              record_model=False):
        '''
        This function is used to perform federated learning with grey box attack.
        :param logger: logger to record the process and write to csv file
        :param record_process: whether to record the process
        :param record_model: whether to record the model
        '''
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
                                                "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                                                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        attacker_success_round = []  # Record the round when attacker succeeds

        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(), DEFAULT_AGR)

        # Print parameters
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info("Dataset is {}".format(DEFAULT_SET))
        logger.info("Member ratio is {}".format(BLACK_BOX_MEMBER_RATE))
        logger.info("cover factor is {},cover dataset size is {}".format(COVER_FACTOR, RESERVED_SAMPLE))

        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        test_loss, test_acc = global_model.test_outcome()

        # Recording and printing
        if record_process:
            acc_recorder.loc[len(acc_recorder)] = (0, "g", test_loss, test_acc, 0)
        logger.info("Global model initiated, loss={}, acc={}".format(test_loss, test_acc))

        # Initialize participants
        participants = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            participants.append(FederatedModel(self.reader, aggregator))
            participants[i].init_participant(global_model, i)
            test_loss, test_acc = participants[i].test_outcome()
            if DEFAULT_AGR == FANG:
                aggregator.agr_model_acquire(global_model) # Fang's aggregator needs to acquire the global model
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, test_loss, test_acc, 0)
            logger.info("Participant {} initiated, loss={}, acc={}".format(i, test_loss, test_acc))

        # Initialize attacker
        attacker = GreyBoxMalicious(self.reader, aggregator)


        for j in range(MAX_EPOCH):
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()
            train_acc_collector = []
            for i in range(NUMBER_OF_PARTICIPANTS):
                # The participants collect the global parameters before training
                participants[i].collect_parameters(global_parameters)
                # The participants calculate local gradients and share to the aggregator
                participants[i].share_gradient()
                # Printing and recording
                # The participants calculate local train loss and train accuracy
                train_loss, train_acc = participants[i].train_loss, participants[i].train_acc
                train_acc_collector.append(train_acc)
                # The participants calculate local test loss and test accuracy
                test_loss, test_acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                logger.info(
                    "Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i,
                                                                                                             test_loss,
                                                                                                             test_acc,
                                                                                                             train_loss,
                                                                                                             train_acc))
            # attacker collects parameter and starts to infer
            attacker.collect_parameters(global_parameters)
            # attacker evaluate the attack result including true member, false member, true non member, false non member
            true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()
            logger.info("true_member {}, false_member {}, true_non_member {}, false_non_member {}".format(true_member,
                                                                                                          false_member,
                                                                                                          true_non_member,
                                                                                                          false_non_member))
            # attacker calculate the attack precision, accuracy, recall
            if true_member and false_member and true_non_member and false_non_member != 0:
                attack_precision = true_member / (true_member + false_member)
                attack_accuracy = (true_member + true_non_member) / (
                        true_member + true_non_member + false_member + false_non_member)
                attack_recall = true_member / (true_member + false_non_member)
            else:
                attack_precision = (true_member + 1) / (true_member + false_member + 1)
                attack_accuracy = (true_member + true_non_member + 1) / (
                        true_member + true_non_member + false_member + false_non_member + 1)
                attack_recall = (true_member + 1) / (true_member + false_non_member + 1)

            # attacker attack
            # attcker train the model within the defined training epoch
            if j < TRAIN_EPOCH:
                attacker.train()
            # attacker perform attack after the training epoch
            else:
                attacker.greybox_attack(cover_factor=COVER_FACTOR, ascent_factor=ASCENT_FACTOR, mislead_factor=MISLEAD_FACTOR)

            # record the aggregator accepted participant's gradient
            if DEFAULT_AGR == FANG:
                logger.info("Selected inputs are from participants number{}".format(aggregator.robust.appearence_list))
                # record success round
                if 5 in aggregator.robust.appearence_list:
                    attacker_success_round.append(j)
                if DEFAULT_AGR == FANG:
                    logger.info("current status {}".format(str(aggregator.robust.status_list)))
            logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision,
                                                                                 attack_recall))

            # attacker evaluate the prediction accuracy of member and non member
            pred_acc_member = attacker.evaluate_member_accuracy().cpu()
            pred_acc_non_member = attacker.evaluate_non_member_accuracy().cpu()
            logger.info("Prediction accuracy, member={}, non-member={}, expected_accuracy={}"
                        .format(pred_acc_member, pred_acc_non_member,
                                BLACK_BOX_MEMBER_RATE * pred_acc_member + (1 - BLACK_BOX_MEMBER_RATE) * (
                                        1 - pred_acc_non_member)))
            attack_recorder.loc[len(attack_recorder)] = (j + 1, \
                                                         attack_accuracy, attack_precision, attack_recall, \
                                                         pred_acc_member, pred_acc_non_member, \
                                                         true_member, false_member, true_non_member, false_non_member)

            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            # Printing and recording
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info(
                "Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc,
                                                                                        train_acc))
        # Printing and recording

        # record best round info
        best_attack_index = attack_recorder["acc"].idxmax()
        best_attack_acc = attack_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_recorder["epoch"][best_attack_index]
        target_model_index = acc_recorder[acc_recorder["epoch"] == best_attack_acc_epoch].index
        target_model_train_acc = acc_recorder["train_accuracy"][target_model_index].values[0]
        target_model_test_acc = acc_recorder["test_accuracy"][target_model_index].values[0]
        member_pred_acc = attack_recorder["pred_acc_member"][best_attack_index]
        non_member_pred_acc = attack_recorder["pred_acc_non_member"][best_attack_index]
        logger.info("attack success round {}, total {}".format(attacker_success_round, len(attacker_success_round)))
        logger.info(
            "Best result: \nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nmember_pred_acc={}\nnon-member_pred_acc={}\nbest_attack_acc_epoch={}\n" \
                .format(best_attack_acc, target_model_train_acc, target_model_test_acc, member_pred_acc, \
                        non_member_pred_acc, best_attack_acc_epoch))
        # record model
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models.csv")

        # record process
        if record_process:
            recorder_suffix = "greybox_misleading"
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                MAX_EPOCH - TRAIN_EPOCH) + recorder_suffix + "optimized_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                   "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                MAX_EPOCH - TRAIN_EPOCH) + recorder_suffix + "optimized_attacker.csv")


    def federated_training_black_box_optimized(self, logger, record_process=True,
                                                   record_model=False):
        '''
        This function is used to perform federated learning with black box attack.
        :param logger: logger to record the process and write to csv file
        :param record_process: whether to record the process
        :param record_model: whether to record the model
        '''
        DEFAULT_AGR = self.args.defense
        ATTACK = self.args.attack
        C = self.args.C 
        T_DELAY = self.args.delay
        DATASET = self.args.dataset
        TRY_TIMES = self.args.cover_times
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
                                                "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                                                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        attacker_success_round = []  # Record the round when attacker succeeds

        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(), DEFAULT_AGR)
        attack_times = 0

        # Print parameters
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info("Attack is {}".format(ATTACK))
        logger.info("Dataset is {}".format(DATASET))
        logger.info("Number of User is {}".format(NUMBER_OF_PARTICIPANTS + NUMBER_OF_ADVERSARY))
        logger.info("C is {}".format(C))
        logger.info("Max Delay is {}".format(T_DELAY))
        logger.info("Try Times is {}".format(TRY_TIMES))
        logger.info("Member ratio is {}".format(BLACK_BOX_MEMBER_RATE))
        logger.info("cover factor is {},cover dataset size is {}".format(COVER_FACTOR, RESERVED_SAMPLE))

        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        test_loss, test_acc = global_model.test_outcome()

        # Recording and printing
        if record_process:
            acc_recorder.loc[len(acc_recorder)] = (0, "g", test_loss, test_acc, 0)
        logger.info("Global model initiated, loss={:.4f}, acc={:.4f}".format(test_loss, test_acc))

        # Initialize participants
        participants = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            participants.append(FederatedModel(self.reader, aggregator))
            participants[i].init_participant(global_model, i)
            test_loss, test_acc = participants[i].test_outcome()
            if DEFAULT_AGR == FANG:
                aggregator.agr_model_acquire(global_model) # Fang's method needs to acquire the global model
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, test_loss, test_acc, 0)
            logger.info("Participant {} initiated, loss={:.4f}, acc={:.4f}".format(i, test_loss, test_acc))

        # Initialize attacker
        attacker = BlackBoxMalicious(self.reader, aggregator)
        # global model history
        global_model_lst = []
        for j in range(MAX_EPOCH):
            # The global model's parameter is shared to each participant before every communication round
            steal_grad_lst = []
            global_parameters = global_model.get_flatten_parameters()
            # save global model for delay
            global_model_lst.append(global_parameters)
            train_acc_collector = []
            # sample  C * (Num_paritcipant + Num_attacker) to participate in training
            random_user_id = random.sample([i for i in range(0, NUMBER_OF_PARTICIPANTS + NUMBER_OF_ADVERSARY)], int(C * (NUMBER_OF_PARTICIPANTS + NUMBER_OF_ADVERSARY)))

            for i in random_user_id:
                # idx = (NUMBER_OF_PARTICIPANTS + NUMBER_OF_ADVERSARY - 1) means choose attacker to participate in training
                if i == NUMBER_OF_PARTICIPANTS + NUMBER_OF_ADVERSARY - 1:
                    continue
                # generate delay
                delay_r = max(0, j - random.randint(0, T_DELAY))
                # The participants collect the global parameters before training
                participants[i].collect_parameters(global_model_lst[delay_r])
                # The participants calculate local gradients and share to the aggregator
                grad = participants[i].share_gradient()
                # TODO: Attack steal the gradient
                steal_grad_lst.append(grad) 
                # The participants calculate local train loss and train accuracy
                train_loss, train_acc = participants[i].train_loss, participants[i].train_acc
                train_acc_collector.append(train_acc)
                # The participants calculate local test loss and test accuracy
                test_loss, test_acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                logger.info(
                    "Epoch {} Participant {}, test_loss={:.4f}, test_acc={:.4f}, train_loss={:.4f}, train_acc={:.4f}".format(j + 1, i,
                                                                                                             test_loss,
                                                                                                             test_acc,
                                                                                                             train_loss,
                                                                                                             train_acc))
            if NUMBER_OF_PARTICIPANTS + NUMBER_OF_ADVERSARY - 1 in random_user_id:
                attack_times += 1
                attacker_delay_r = max(0, j - random.randint(0, T_DELAY))
                # attacker collects parameter and starts to infer
                attacker.collect_parameters(global_model_lst[attacker_delay_r])
            # attacker evaluate the attack result including true member, false member, true non member, false non member
            true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()
            logger.info(
                "true_member {}, false_member {}, true_non_member {}, false_non_member {}".format(true_member,
                                                                                                  false_member,
                                                                                                  true_non_member,
                                                                                                  false_non_member))
            # attacker calculate the attack precision, accuracy, recall
            if true_member and false_member and true_non_member and false_non_member != 0:
                attack_precision = true_member / (true_member + false_member)
                attack_accuracy = (true_member + true_non_member) / (
                        true_member + true_non_member + false_member + false_non_member)
                attack_recall = true_member / (true_member + false_non_member)
            else:
                attack_precision = (true_member + 1) / (true_member + false_member + 1)
                attack_accuracy = (true_member + true_non_member + 1) / (
                        true_member + true_non_member + false_member + false_non_member + 1)
                attack_recall = (true_member + 1) / (true_member + false_non_member + 1)

            if NUMBER_OF_PARTICIPANTS + NUMBER_OF_ADVERSARY - 1 in random_user_id:
                print(f'Attack: {ATTACK}')
                # attacker attack
                # attcker train the model within the defined training epoch
                if j < TRAIN_EPOCH:
                    attacker.train()
                # attacker attack the model within the defined attack epoch
                else:
                    if ATTACK == 'angle':
                        print('angle attack')
                        attacker.blackbox_attack_angle(cover_factor=COVER_FACTOR, grad_honest=steal_grad_lst, try_times=TRY_TIMES)
                    elif ATTACK == 'unit':
                        print('unitnorm attack')
                        attacker.blackbox_attack_unit(cover_factor=COVER_FACTOR, grad_honest=steal_grad_lst)
                    elif ATTACK == 'norm':
                        print('norm attack')
                        attacker.blackbox_attack_norm(cover_factor=COVER_FACTOR, grad_honest=steal_grad_lst)
                    else:
                        print('Origin attack')
                        attacker.blackbox_attack_origin(cover_factor=COVER_FACTOR)

            # record the aggregator accepted participant's gradient
            if DEFAULT_AGR is FANG:
                logger.info("Selected inputs are from participants number{}".format(aggregator.robust.appearence_list))
                #record the attacker success round
                if 5 in aggregator.robust.appearence_list:
                    attacker_success_round.append(j)
                logger.info("current status {}".format(str(aggregator.robust.status_list)))

            # attacker evaluate the prediction accuracy of member and non member
            logger.info("Attack accuracy = {:.4f}, Precision = {:.4f}, Recall={:.4f}".format(attack_accuracy, attack_precision,
                                                                                 attack_recall))
            pred_acc_member = attacker.evaluate_member_accuracy()
            pred_acc_non_member = attacker.evaluate_non_member_accuracy()
            logger.info("Prediction accuracy, member={:.4f}, non-member={:.4f}, expected_accuracy={:.4f}"
                        .format(pred_acc_member, pred_acc_non_member,
                                BLACK_BOX_MEMBER_RATE * pred_acc_member + (1 - BLACK_BOX_MEMBER_RATE) * (
                                            1 - pred_acc_non_member)))
            # record the attack result
            attack_recorder.loc[len(attack_recorder)] = (j + 1, \
                                                         attack_accuracy, attack_precision, attack_recall, \
                                                         pred_acc_member, pred_acc_non_member, \
                                                         true_member, false_member, true_non_member, false_non_member)
            # Global model collects the aggregated gradient
            global_model.apply_gradient()

            #Global model calculate the test loss and test accuracy
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info(
                "Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc,
                                                                                        train_acc))
        # Printing and recording
        # record best round info
        best_attack_index = attack_recorder["acc"].idxmax()
        best_attack_acc = attack_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_recorder["epoch"][best_attack_index]
        target_model_index = acc_recorder[acc_recorder["epoch"] == best_attack_acc_epoch].index
        target_model_train_acc = acc_recorder["train_accuracy"][target_model_index].values[0]
        target_model_test_acc = acc_recorder["test_accuracy"][target_model_index].values[0]
        member_pred_acc = attack_recorder["pred_acc_member"][best_attack_index]
        non_member_pred_acc = attack_recorder["pred_acc_non_member"][best_attack_index]
        logger.info("attack success round {}, total {}".format(attacker_success_round, len(attacker_success_round)))

        logger.info(
            "Best result: \nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nmember_pred_acc={}\nnon-member_pred_acc={}\nbest_attack_acc_epoch={}\nattacker_participate_times={}\n" \
                .format(best_attack_acc, target_model_train_acc, target_model_test_acc, member_pred_acc, \
                        non_member_pred_acc, best_attack_acc_epoch, attack_times))

        # record the model
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DATASET + "Federated_Models.csv")

        # record process
        if record_process:
            recorder_suffix = "blackbox"
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + DATASET + str(DEFAULT_AGR) + str(ATTACK)\
                                + "User" + str(NUMBER_OF_PARTICIPANTS+NUMBER_OF_ADVERSARY) + "C" + str(C) + "Delay" + str(T_DELAY) \
                                + str(DATA_DISTRIBUTION) + "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                                MAX_EPOCH - TRAIN_EPOCH)+ recorder_suffix + "optimized_model" + TIME_STAMP + ".csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + DATASET + str(DEFAULT_AGR) + str(ATTACK)\
                                + "User" + str(NUMBER_OF_PARTICIPANTS+NUMBER_OF_ADVERSARY) + "C" + str(C) + "Delay" + str(T_DELAY) \
                                + str(DATA_DISTRIBUTION) + "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                                MAX_EPOCH - TRAIN_EPOCH)+ recorder_suffix + "optimized_attacker" + TIME_STAMP + ".csv")