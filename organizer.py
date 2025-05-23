from models import *
from constants import *
import pandas as pd
import numpy as np
import os, random
from data_reader import DataReader
import argparse
import math
import time

def calculate_roc_metrics(confidences, labels):
    """
    Calculate ROC curve metrics from confidence scores and labels
    Args:
        confidences: array of confidence scores
        labels: array of ground truth labels (1/0)
    Returns:
        fpr: list of false positive rates
        tpr: list of true positive rates
        thresholds: list of thresholds used
    """
    # 按置信度降序排序
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_confidences = confidences[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # 初始化统计量
    tpr = []
    fpr = []
    thresholds = []
    
    # 累积TP/FP统计
    total_p = np.sum(sorted_labels == 1)
    total_n = np.sum(sorted_labels == 0)
    
    fp = 0
    tp = 0
    
    prev_confidence = None
    for i in range(len(sorted_confidences)):
        if sorted_confidences[i] != prev_confidence:
            # 保存当前阈值的结果
            tpr.append(tp / total_p if total_p > 0 else 0)
            fpr.append(fp / total_n if total_n > 0 else 0)
            thresholds.append(sorted_confidences[i])
            prev_confidence = sorted_confidences[i]
        
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
    
    # 添加最后一个点
    tpr.append(tp / total_p if total_p > 0 else 0)
    fpr.append(fp / total_n if total_n > 0 else 0)
    thresholds.append(0.0)
    
    return fpr, tpr, thresholds

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
        choices=['norm', 'unit', 'angle', 'cos', 'origin', 'gradient_ascent', 'adaptive'],
        default='angle'
    )

    parser.add_argument(
        "-d",
        "--defense",
        help="normal users defense type",
        required=True,
        choices=['Fang', 'Median', 'Trim', 'Krum', 'Multi-Krum', 'Deepsight', 'Rflbat', 'Flame', 'Foolsgold', 'None', 'Angle-Median', 'Angle-Trim',
                 'Deepsight', 'Topk', 'Differential-Privacy'],
    )
    
    parser.add_argument(
        "-dt",
        "--dataset",
        help="dataset",
        required=True,
        # choices=['CIFAR10', 'Location30', 'Purchase100', 'MNIST', 'FASHION_MNIST', 'Texas100', 'CIFAR100'],
        # default='CIFAR10'
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

    parser.add_argument(
        "--Number_client",
        help="Number of clients",
        default=10,
        type=int
    )

    parser.add_argument(
        "--Number_malicious",
        help="Number of malicious clients",
        default=1,
        type=int
    )
    
    parser.add_argument(
        "--device",
        help="device index",
        type=int,
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
        "--max_epoch",
        help="Number of epoches to run totally",
        default=800,
        type=int
    )

    parser.add_argument(
        "--train_epoch",
        help="Number of epoches for attacker to normal training",
        default=0,
        type=int
    )

    parser.add_argument(
        "--number_attack_samples",
        help="Number of attack sample including member and non-member",
        default=300,
        type=int
    )
    
    parser.add_argument(
        "--output_dir",
        help="log output direction.",
        default='./log/',
        type=str
    )
    
    parser.add_argument(
        "--bias",
        help="noniid bias",
        default=0.5,
        type=float
    )

    args = parser.parse_args()
    return args

def check_convergence(loss_values, threshold=1e-3):
    """
    通过计算损失函数的斜率判断模型是否收敛

    参数:
    loss_values (list): 损失函数值列表
    threshold (float): 判定收敛的斜率阈值

    返回:
    bool: 模型是否收敛
    """

    if len(loss_values) < 2:
        return False

    # 计算损失函数的变化率 (斜率)
    slopes = np.diff(loss_values)

    # 计算斜率的绝对值的平均值
    avg_slope = np.mean(np.abs(slopes))

    # 如果平均斜率低于阈值，则认为模型收敛
    return avg_slope < threshold
    
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
        self.MAX_EPOCH = self.args.max_epoch
        self.TRAIN_EPOCH = self.args.train_epoch
        self.DEVICE =  torch.device(f"cuda:{self.args.device}" if torch.cuda.is_available() else "cpu")
        self.set_random_seed()
        self.reader = DataReader(data_set=self.args.dataset, number_clients=self.args.Number_client - self.args.Number_malicious, 
                                data_distribution=self.DATA_DISTRIBUTION, reserved=self.args.number_attack_samples,
                                device=self.DEVICE, noniid_bias=self.args.bias)
        self.target = TargetModel(self.reader, participant_index=0, device=self.DEVICE, model=self.args.dataset)
        print(f'target model: {self.target.model}')
    
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
            if j < self.TRAIN_EPOCH:
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
        EXPERIMENTAL_DATA_DIRECTORY = self.args.output_dir
        NUMBER_OF_ADVERSARY = self.args.Number_malicious
        NUMBER_OF_PARTICIPANTS = self.args.Number_client - NUMBER_OF_ADVERSARY
        NUMBER_OF_ATTACK_SAMPLES = self.args.number_attack_samples
        RESERVED_SAMPLE = NUMBER_OF_ATTACK_SAMPLES
        ATTACK_ROUND = []

        attack_time_list = []
        labels_list = []
        confidences_list = []

        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
                                                "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                                                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        attacker_success_round = []  # Record the round when attacker succeeds

        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(), self.DEVICE, DEFAULT_AGR)
        attack_times = 0
        start_attack_flag = False; start_attack_epoch = 0

        # Print parameters
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info("Attack is {}".format(ATTACK))
        logger.info("Dataset is {}".format(DATASET))
        logger.info("Number of User is {}".format(NUMBER_OF_PARTICIPANTS + NUMBER_OF_ADVERSARY))
        logger.info("Number of Client is {}".format(NUMBER_OF_PARTICIPANTS))
        logger.info("Number of Melitious Client is {}".format(NUMBER_OF_ADVERSARY))
        logger.info("Number of Attack Samples is {}".format(NUMBER_OF_ATTACK_SAMPLES))
        logger.info("C is {}".format(C))
        logger.info("Max Delay is {}".format(T_DELAY))
        logger.info("Member ratio is {}".format(BLACK_BOX_MEMBER_RATE))
        logger.info("cover factor is {},cover dataset size is {}".format(COVER_FACTOR, RESERVED_SAMPLE))

        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator, DATASET, self.DEVICE)
        global_model.init_global_model()
        test_loss, test_acc = global_model.test_outcome()

        # Recording and printing
        if record_process:
            acc_recorder.loc[len(acc_recorder)] = (0, "g", test_loss, test_acc, 0)
        logger.info("Global model initiated, loss={:.4f}, acc={:.4f}".format(test_loss, test_acc))

        # Initialize participants
        participants = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            participants.append(FederatedModel(self.reader, aggregator, DATASET, self.DEVICE))
            participants[i].init_participant(global_model, i)
            test_loss, test_acc = participants[i].test_outcome()
            if DEFAULT_AGR == FANG:
                aggregator.agr_model_acquire(global_model) # Fang's method needs to acquire the global model
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, test_loss, test_acc, 0)
            logger.info("Participant {} initiated, loss={:.4f}, acc={:.4f}".format(i, test_loss, test_acc))

        # Initialize attackers
        attacker = BlackBoxMalicious(self.reader, aggregator, DATASET, self.DEVICE, NUMBER_OF_ATTACK_SAMPLES)

        # global model history
        global_model_lst = []
        global_model_loss_lst = []
        global_model_trainAcc_lst = []
        for j in range(self.MAX_EPOCH):
            # The global model's parameter is shared to each participant before every communication round
            steal_grad_lst = []
            global_parameters = global_model.get_flatten_parameters()
            # save global model for delay
            global_model_lst.append(global_parameters)
            train_acc_collector = []
            train_loss_collector = []
            # sample  C * (Num_paritcipant + Num_attacker) to participate in training
            random_user_id = random.sample([i for i in range(0, NUMBER_OF_PARTICIPANTS + NUMBER_OF_ADVERSARY)], math.ceil(C * (NUMBER_OF_PARTICIPANTS + NUMBER_OF_ADVERSARY)))

            for i in random_user_id:
                # idx > (NUMBER_OF_PARTICIPANTS + NUMBER_OF_ADVERSARY - 1) means choose attacker to participate in training
                if i > NUMBER_OF_PARTICIPANTS - 1:
                    continue
                # generate delay
                delay_r = max(0, j - random.randint(0, T_DELAY))
                # The participants collect the global parameters before training
                participants[i].collect_parameters(global_model_lst[delay_r])
                # The participants calculate local gradients and share to the aggregator
                grad = participants[i].share_gradient()
                # NOTE: Attack steal the gradient
                steal_grad_lst.append(grad) 
                # The participants calculate local train loss and train accuracy
                train_loss, train_acc = participants[i].train_loss, participants[i].train_acc
                train_acc_collector.append(train_acc)
                train_loss_collector.append(train_loss)
                # The participants calculate local test loss and test accuracy
                test_loss, test_acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                logger.info(
                    "Epoch {} Participant {} use {}, test_loss={:.4f}, test_acc={:.4f}, train_loss={:.4f}, train_acc={:.4f}".format(j + 1, i,
                                                                                                             delay_r,
                                                                                                             test_loss,
                                                                                                             test_acc,
                                                                                                             train_loss,
                                                                                                             train_acc))
            Number_selected_malicious_client = 0
            for i in range(0, NUMBER_OF_ADVERSARY):
                if NUMBER_OF_PARTICIPANTS + i in random_user_id:
                    attack_times += 1
                    Number_selected_malicious_client += 1
            
            if Number_selected_malicious_client != 0:
                attacker_delay_r = max(0, j - random.randint(0, T_DELAY))
                # attacker collects parameter and starts to infer
                attacker.collect_parameters(global_model_lst[attacker_delay_r])
            # attacker evaluate the attack result including true member, false member, true non member, false non member
            true_member, false_member, true_non_member, false_non_member, labels, confidences = attacker.evaluate_attack_result()
            logger.info(
                "true_member {}, false_member {}, true_non_member {}, false_non_member {}".format(true_member,
                                                                                                  false_member,
                                                                                                  true_non_member,
                                                                                                  false_non_member))
            labels_list.append(labels)
            confidences_list.append(confidences)
            
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

            if Number_selected_malicious_client != 0:
                # attacker attack
                # attcker train the model within the defined training epoch
                if j < self.TRAIN_EPOCH:
                    passive_start_time = time.time()
                    attacker.train()
                    passive_end_time = time.time()
                    logger.info(f'passive_time = {passive_end_time-passive_start_time}')
                # attacker attack the model within the defined attack epoch
                else:
                    ATTACK_ROUND.append(j)
                    # 记录attack的时间
                    attack_start_time = time.time()
                    if ATTACK == 'angle':
                        print('angle attack')
                        # attacker.blackbox_attack_angle_new(num_malicious=Number_selected_malicious_client, 
                        attacker.blackbox_attack_angle(num_malicious=Number_selected_malicious_client, 
                                                        cover_factor=1.0, grad_honest=steal_grad_lst, 
                                                        logger=logger)   
                    elif ATTACK == 'unit':
                        print('angle_random attack')
                        # print('unitnorm attack')
                        attacker.blackbox_attack_angle_random(num_malicious=Number_selected_malicious_client, 
                                                        cover_factor=1.0, grad_honest=steal_grad_lst, 
                                                        logger=logger)
                    elif ATTACK == 'norm':
                        print('norm attack')
                        # attacker.blackbox_attack_norm(num_malicious=Number_selected_malicious_client,
                        attacker.blackbox_attack_origin_norm(num_malicious=Number_selected_malicious_client,
                                                            cover_factor=COVER_FACTOR, grad_honest=steal_grad_lst, 
                                                            logger=logger)
                    elif ATTACK == 'gradient_ascent':
                        print('gradient ascent')
                        attacker.blackbox_attack_ascent()
                        
                    elif ATTACK == 'adaptive':
                        print('adaptive attack')
                        attacker.blackbox_attack_adaptive(num_malicious=Number_selected_malicious_client, 
                                                        cover_factor=COVER_FACTOR, grad_honest=steal_grad_lst, 
                                                        logger=logger)
                    
                        
                    else:
                        print('Origin attack')
                        attacker.blackbox_attack_origin(num_malicious=Number_selected_malicious_client, 
                                                        cover_factor=COVER_FACTOR)
                    attack_end_time = time.time()
                    attack_time_list.append(attack_end_time - attack_start_time)
                    logger.info(f'Epoch {j} attack time: {attack_end_time - attack_start_time}')
                    logger.info(f'Average attack time: {sum(attack_time_list) / len(attack_time_list)}')
                    
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
            train_loss = torch.mean(torch.tensor(train_loss_collector)).item()
            global_model_loss_lst.append(train_loss)
            global_model_trainAcc_lst.append(train_acc)
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info(
                "Epoch {} Global model, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, test_loss, test_acc,
                                                                                                        train_loss, train_acc))
            
            # check whether the attacker can start attack(whether the global model is converaged)
            if not start_attack_flag and len(global_model_loss_lst) > 2 and check_convergence(global_model_trainAcc_lst[-3:]):
                start_attack_flag = True
                logger.info(f"Start attack at {j}")
                start_attack_epoch = j
                
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
            "Best result: \nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nmember_pred_acc={}\nnon-member_pred_acc={}\nbest_attack_acc_epoch={}\nattacker_participate_times={}\nstart_attack_epoch={}\n" 
                .format(best_attack_acc, target_model_train_acc, target_model_test_acc, member_pred_acc, \
                        non_member_pred_acc, best_attack_acc_epoch, attack_times//NUMBER_OF_ADVERSARY, start_attack_epoch))

        logger.info(f'attack round={ATTACK_ROUND}')
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
                                + str(self.DATA_DISTRIBUTION) + "TrainEpoch" + str(self.TRAIN_EPOCH) + "AttackEpoch" + str(
                                self.MAX_EPOCH - self.TRAIN_EPOCH)+ recorder_suffix + "optimized_model" + TIME_STAMP + ".csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + DATASET + str(DEFAULT_AGR) + str(ATTACK)\
                                + "User" + str(NUMBER_OF_PARTICIPANTS+NUMBER_OF_ADVERSARY) + "C" + str(C) + "Delay" + str(T_DELAY) \
                                + str(self.DATA_DISTRIBUTION) + "TrainEpoch" + str(self.TRAIN_EPOCH) + "AttackEpoch" + str(
                                self.MAX_EPOCH - self.TRAIN_EPOCH)+ recorder_suffix + "optimized_attacker" + TIME_STAMP + ".csv")
            
        # 绘制best attack的roc曲线
        labels = labels_list[best_attack_index]
        confidences = confidences_list[best_attack_index]
        
        fpr, tpr, thresholds = calculate_roc_metrics(confidences, labels)
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fpr, tpr, label='FedPoisonMIA')
        # plt.xlim([0, 0.05])  # 放大低FPR区域
        plt.xlabel('False Positive Rate')   
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        # plt.show()
        
        
        # 计算TPR@0.1%FPR
        target_fpr = 0.001
        # 找到第一个FPR >= target_fpr的索引
        idx = np.where(np.array(fpr) >= target_fpr)[0][0]
        tpr_at_low_fpr = tpr[idx]
        print(f"TPR@0.1%FPR: {tpr_at_low_fpr:.4f}")
        
        # 将target_fpr对应的tpr画在图上
        plt.scatter(target_fpr, tpr_at_low_fpr, c='r', marker='o')
        plt.text(target_fpr, tpr_at_low_fpr, f'({target_fpr:.4f}, {tpr_at_low_fpr:.4f})', ha='right')
        plt.savefig(f'./roc_{self.args.defense}_{self.args.attack}.png')
        
        
        
