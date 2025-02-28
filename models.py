import logging
import os
import random
import sys
from aggregator import *
from data_reader import DataReader
import numpy as np
from torch import nn
from copy import deepcopy
from scipy.optimize import minimize

def make_logger(name, save_dir, save_filename):
    """
    Make a logger to record the training process
    :param name: logger name
    :param save_dir: the directory to save the log file
    :param save_filename: the filename to save the log file
    :return: logger
    """
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt=DATE_FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, save_filename + ".txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def select_by_threshold(to_share: torch.Tensor, select_fraction: float, device, select_threshold: float = 1):
    """
    Apply the privacy-preserving method following selection-by-threshold approach
    :param to_share: the tensor to share
    :param select_fraction: the fraction of the tensor to share
    :param select_threshold: the threshold to select the tensor
    :return: the shared tensor and the indices of the selected tensor
    """
    threshold_count = round(to_share.size(0) * select_threshold)
    selection_count = round(to_share.size(0) * select_fraction)
    indices = to_share.topk(threshold_count).indices
    perm = torch.randperm(threshold_count).to(device)
    indices = indices[perm[:selection_count]]
    rei = torch.zeros(to_share.size()).to(device)
    rei[indices] = to_share[indices].to(device)
    to_share = rei.to(device)
    return to_share, indices


class ModelLocation30(torch.nn.Module):
    """
    The model to handel Location100 dataset
    """

    def __init__(self):
        super(ModelLocation30, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(446, 512),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 30),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out

class ModelPurchase100(torch.nn.Module):
    """
    The model to handel Location100 dataset
    """

    def __init__(self):
        super(ModelPurchase100, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(600, 512),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 100),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out

class ModelTexas100(torch.nn.Module):
    """
    The model to handel Location100 dataset
    """

    def __init__(self):
        super(ModelTexas100, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(6169, 512),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 100),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out
    
# 定义 ResBlock 和 ResNet20 类
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels) if bn else nn.Identity()
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet20(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), pooling_size=8, output_shape=10, bn=True):
        super(ResNet20, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16) if bn else nn.Identity()
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(16, 16, 3, stride=1, bn=bn)
        self.layer2 = self._make_layer(16, 32, 3, stride=2, bn=bn)
        self.layer3 = self._make_layer(32, 64, 3, stride=2, bn=bn)
        self.avgpool = nn.AvgPool2d(pooling_size)
        self.fc = nn.Linear(256, output_shape) #! covid19时需要使用256，其余64即可

    def _make_layer(self, in_channels, out_channels, blocks, stride, bn):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, bn))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1, bn=bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TargetModel:
    """
    The model to attack against, the target for attacking
    """

    def __init__(self, data_reader: DataReader, model, device, participant_index=0):
        self.DEVICE = device
        # initialize the model
        print(f'model: {model}')
        if model == LOCATION30:
            self.model = ModelLocation30()
        elif model == PURCHASE100:
            self.model = ModelPurchase100()
        elif model == TEXAS100:
            self.model = ModelTexas100()
        elif model == CIFAR10 or model == CINIC10 or model == SVHN:
            self.model = ResNet20()
        elif model == CIFAR100:
            self.model = ResNet20(output_shape=100)
        elif model == MNIST:
            self.model = ResNet20(input_shape=(1, 28, 28), pooling_size=7, output_shape=10)
        elif model == FASHION_MNIST:
            self.model = ResNet20(input_shape=(1, 28, 28), pooling_size=7, output_shape=10)
        elif model == GTSRB:
            self.model = ResNet20(output_shape=43)
        elif model == SUN397:
            self.model = ResNet20(output_shape=397)
        elif model == STL10:
            self.model = ResNet20(input_shape=(3, 96, 96), pooling_size=16, output_shape=10)
        elif model == FER2013:
            self.model = ResNet20(input_shape=(3, 48, 48), pooling_size=8, output_shape=7)
        elif model == COVID19:
            # self.model = ResNet20(input_shape=(3, 224, 224), pooling_size=28, output_shape=4)
            self.model = ResNet20(input_shape=(3, 128, 128), pooling_size=16, output_shape=4)   
        else:
            raise NotImplementedError("Model not supported")
        self.model = self.model.to(self.DEVICE)

        # initialize the data
        self.test_set = None
        self.train_set = None
        self.data_reader = data_reader
        self.participant_index = participant_index
        self.load_data()

        # initialize the loss function and optimizer
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.DEVICE)
        # learning rate keeps default value 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters())
        # self.optimizer = torch.optim.Adamax(self.model.parameters())
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

        # Initialize recorder
        self.train_loss = -1
        self.train_acc = -1

        # fc layer parameters 


    def load_data(self):
        """
        Load batch indices from the data reader
        :return: None
        """
        self.train_set = self.data_reader.get_train_set(self.participant_index).to(self.DEVICE)
        self.test_set = self.data_reader.get_test_set(self.participant_index).to(self.DEVICE)
        
        
        # print('**************************************')
        # print(f'par{self.participant_index}')
        # # 统计trian_set对应下标，中各label的数量
        # labels = {i: 0 for i in range(self.data_reader.num_class)}
        # for batch in self.train_set:
        #     for indice in batch:
        #         label = self.data_reader.labels[indice] #! self.data_reader.labels[indice]
        #         labels[int(label)] += 1
        # for i in range(10):
        #     print(f'label {i}: {labels[i]}')
        # print('**************************************')
        


    def normal_epoch(self, print_progress=False, by_batch=BATCH_TRAINING):
        """
        Train a normal epoch with the given dataset
        :param print_progress: if print the training progress or not
        :param by_batch: True to train by batch, False otherwise
        :return: the training accuracy and the training loss value
        """
        train_loss = 0
        train_acc = 0
        batch_counter = 0
        if by_batch:
            for batch_indices in self.train_set:
                batch_counter += 1
                if print_progress and batch_counter % 100 == 0:
                    print("Currently training for batch {}, overall {} batches"
                          .format(batch_counter, self.train_set.size(0)))
                batch_x, batch_y = self.data_reader.get_batch(batch_indices)
                batch_x = batch_x.to(self.DEVICE)
                batch_y = batch_y.to(self.DEVICE)
                out = self.model(batch_x).to(self.DEVICE)
                batch_loss = self.loss_function(out, batch_y)
                train_loss += batch_loss.item()
                prediction = torch.max(out, 1).indices.to(self.DEVICE)
                batch_acc = (prediction == batch_y).sum()
                train_acc += batch_acc.item()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
        self.train_acc = train_acc / (self.train_set.flatten().size(0))
        self.train_loss = train_loss / (self.train_set.flatten().size(0))
        if print_progress:
            print("Epoch complete for participant {}, train acc = {}, train loss = {}"
                  .format(self.participant_index, train_acc, train_loss))
        return self.train_loss, self.train_acc

    def test_outcome(self, by_batch=BATCH_TRAINING):
        """
        Test through the test set to get loss value and accuracy
        :param by_batch: True to test by batch, False otherwise
        :return: the test accuracy and test loss value
        """
        test_loss = 0
        test_acc = 0
        if by_batch:
            for batch_indices in self.test_set:
                batch_x, batch_y = self.data_reader.get_batch(batch_indices)
                batch_x = batch_x.to(self.DEVICE)
                batch_y = batch_y.to(self.DEVICE)
                with torch.no_grad():
                    # print(f'batch x shape: {batch_x.shape}')
                    # batch x shape: torch.Size([64, 3, 32, 32])
                    out = self.model(batch_x).to(self.DEVICE)
                    batch_loss = self.loss_function(out, batch_y).to(self.DEVICE)
                    test_loss += batch_loss.item()
                    prediction = torch.max(out, 1).indices.to(self.DEVICE)
                    batch_acc = (prediction == batch_y).sum().to(self.DEVICE)
                    test_acc += batch_acc.item()
        test_acc = test_acc / (self.test_set.flatten().size(0))
        test_loss = test_loss / (self.test_set.flatten().size(0))
        return test_loss, test_acc

    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0).to(self.DEVICE)
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                out = torch.cat([out, parameter.flatten()]).to(self.DEVICE)
        return out

    def get_fc_flatten_parameters(self):
        """
        Return the flatten parameter of the last lineat layer
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0).to(self.DEVICE)
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if 'fc' in name:
                    out = torch.cat([out, parameter.flatten()]).to(self.DEVICE)
        return out

    def load_parameters(self, parameters: torch.Tensor):
        """
        Load parameters to the current model using the given flatten parameters
        :param parameters: The flatten parameter to load
        :return: None
        """
        start_index = 0
        for param in self.model.parameters():
            length = len(param.flatten())
            to_load = parameters[start_index: start_index + length].to(self.DEVICE)
            to_load = to_load.reshape(param.size()).to(self.DEVICE)
            with torch.no_grad():
                param.copy_(to_load).to(self.DEVICE)
            start_index += length

    def get_epoch_gradient(self, apply_gradient=True):
        """
        Get the gradient for the current epoch
        :param apply_gradient: if apply the gradient or not
        :return: the tensor contains the gradient
        """
        cache = self.get_flatten_parameters().to(self.DEVICE)
        self.normal_epoch()
        gradient = self.get_flatten_parameters() - cache.to(self.DEVICE)
        if not apply_gradient:
            self.load_parameters(cache)
        return gradient

    def init_parameters(self, mode=INIT_MODE):
        """
        Initialize the parameters according to given mode
        :param mode: the mode to init with
        :return: None
        """
        if mode == PYTORCH_INIT:
            return
        else:
            raise ValueError("Invalid initialization mode")

    def test_gradients(self, gradient: torch.Tensor):
        """
        Make use of the given gradients to run a test, then revert back to the previous status
        :param gradient: the gradient to apply
        :return: the loss and accuracy of the test
        """
        cache = self.get_flatten_parameters()
        test_param = cache + gradient
        self.load_parameters(test_param)
        loss, acc = self.test_outcome()
        self.load_parameters(cache)
        return loss, acc

class FederatedModel(TargetModel):
    """
    Representing the class of federated learning members
    """
    def __init__(self, reader: DataReader, aggregator: Aggregator, model, device, participant_index=0):
        """
        Initialize the federated model
        :param reader: initialize the data reader
        :param aggregator: initialize the aggregator
        :param participant_index: the index of the participant
        """
        super(FederatedModel, self).__init__(reader, model, device, participant_index)
        self.DEVICE = device
        self.aggregator = aggregator

    def init_global_model(self):
        """
        Initialize the current model as the global model
        :return: None
        """
        self.init_parameters()
        self.test_set = self.data_reader.test_set.to(self.DEVICE)
        self.train_set = None

    def init_participant(self, global_model: TargetModel, participant_index):
        """
        Initialize the current model as a participant
        :return: None
        """
        self.participant_index = participant_index
        self.load_parameters(global_model.get_flatten_parameters())
        self.load_data()

    def share_gradient(self):
        """
        Participants share gradient to the aggregator
        :return: None
        """
        gradient = self.get_epoch_gradient()
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices=indices, source=self.participant_index)
        return gradient

    def apply_gradient(self):
        """
        Global model applies the gradient
        :return: None
        """
        parameters = self.get_flatten_parameters()
        parameters += self.aggregator.get_outcome(reset=True)
        self.load_parameters(parameters)

    def collect_parameters(self, parameter: torch.Tensor):
        """
        Participants collect parameters from the global model
        :param parameter: the parameters shared by the global model
        :return: None
        """
        to_load = self.get_flatten_parameters().to(self.DEVICE)
        parameter, indices = select_by_threshold(parameter, PARAMETER_EXCHANGE_RATE, self.DEVICE, PARAMETER_SAMPLE_THRESHOLD)
        to_load[indices] = parameter[indices]
        self.load_parameters(to_load)

class BlackBoxMalicious(FederatedModel):
    """
    Representing the malicious participant trying to perform a black-box membership inference attack
    """

    def __init__(self, reader: DataReader, aggregator: Aggregator, model, device, number_attack_sample):
        """
        Initialize the black-box malicious participant
        :param reader: Reader to read the data
        :param aggregator: Global aggregator
        """
        super(BlackBoxMalicious, self).__init__(reader, aggregator, model, device)
        self.DEVICE = device
        self.attack_samples, self.members, self.non_members = reader.get_black_box_batch(attack_batch_size=number_attack_sample)
        self.member_count = 0
        self.batch_x, self.batch_y = self.data_reader.get_batch(self.attack_samples)
        self.shuffled_y = self.shuffle_label(self.batch_y)
        self.best_cover_set = None
        self.best_AGREvader_grad = None
        for i in self.attack_samples:
            if i in reader.train_set:
                self.member_count += 1
        print(f'attack sample: {len(self.attack_samples)}')
        print(f'member count: {self.member_count}')


    def shuffle_label(self, ground_truth: torch.Tensor):
        """
        Shuffle the labels of the given ground truth
        :param ground_truth: The ground truth to shuffled data
        :return: Shuffled labels
        """
        result = ground_truth[torch.randperm(ground_truth.size()[0])]
        for i in range(ground_truth.size()[0]):
            while result[i].eq(ground_truth[i]):
                result[i] = torch.randint(ground_truth.max(), (1,))
        return result

    def train(self):
        """
        Normal training process for the black-box malicious participant
        :return: Gradient of the current round
        """
        cache = self.get_flatten_parameters()
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices)
        return gradient


    def blackbox_attack_origin(self,num_malicious=1, cover_factor = 0,batch_size = BATCH_SIZE):
        """
        Optimized shuffle label attack
        :param cover_factor: Cover factor of the gradient of cover samples
        :param batch_size: The size of the batch
        :return: The malicious gradient covered by gradient of cover samples for current round
        """

        cache = self.get_flatten_parameters()
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.shuffled_y) # compute loss with shuffled labels
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        self.load_parameters(cache)
        # !! cover sample
        candidate_cover_samples = self.data_reader.reserve_set #cover samples
        # rand_indices = torch.randperm(candidate_cover_samples.size(0))
        # selected_indices = rand_indices[:320]
        # cover_samples = candidate_cover_samples[selected_indices]
        cover_samples = candidate_cover_samples
        
        i = 0
        while i * batch_size < len(cover_samples):
            batch_index = cover_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        cover_gradient = self.get_flatten_parameters() - cache

        if RESERVED_SAMPLE != 0:
            # 这里跟论文里提到的公式似乎不太一样
            gradient, indices = select_by_threshold(cover_gradient*cover_factor+gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD) # computed the malicious gradient
        else:
            gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices, sample_count=num_malicious)

        return gradient

    def blackbox_attack_origin_norm(self,num_malicious=1, cover_factor = 0,batch_size = BATCH_SIZE, grad_honest = None, logger=None):
        """
        Optimized shuffle label attack
        :param cover_factor: Cover factor of the gradient of cover samples
        :param batch_size: The size of the batch
        :return: The malicious gradient covered by gradient of cover samples for current round
        """
        # get max_honest_diff
        max_honest_diff = 0.0
        for i in range(len(grad_honest)):
            for j in range(i+1, len(grad_honest)):
                max_honest_diff = max(max_honest_diff, torch.norm(grad_honest[i] - grad_honest[j]))
                
        cache = self.get_flatten_parameters()
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.shuffled_y) # compute loss with shuffled labels
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        vic_grad = self.get_flatten_parameters() - cache
        self.load_parameters(cache)
        
        # !! cover sample
        candidate_cover_samples = self.data_reader.reserve_set #cover samples
        cover_samples = candidate_cover_samples
        
        i = 0
        while i * batch_size < len(cover_samples):
            batch_index = cover_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        cov_grad = self.get_flatten_parameters() - cache
        self.load_parameters(cache)
        
        agr_grad = cov_grad * cover_factor + vic_grad
                
        max_diff = 0.0
        
        for k in range(len(grad_honest)):
            max_diff = max(torch.norm(agr_grad - grad_honest[k]), max_diff)

        logger.info(f'max_diff: {max_diff}, max_honest_diff: {max_honest_diff}')
        if max_diff <= max_honest_diff :
            self.aggregator.collect(agr_grad, sample_count=num_malicious)
            logger.info('use agr_grad')
            return agr_grad
        else: 
            self.aggregator.collect(cov_grad, sample_count=num_malicious)
            logger.info('use vic_grad')
            return vic_grad    
    
    def blackbox_attack_norm(self,num_malicious=1, cover_factor = 0,batch_size = BATCH_SIZE, grad_honest = None, logger=None):
        """
        Optimized shuffle label attack
        :param cover_factor: Cover factor of the gradient of cover samples
        :param batch_size: The size of the batch
        :return: The malicious gradient covered by gradient of cover samples for current round
        """
        # assert len(grad_honest) == NUMBER_OF_PARTICIPANTS, "honest gradient have not fully collected!"
        # 获取max_honest_diff
        max_honest_diff = 0.0
        for i in range(len(grad_honest)):
            for j in range(i+1, len(grad_honest)):
                max_honest_diff = max(max_honest_diff, torch.norm(grad_honest[i] - grad_honest[j]))
                
        # for grad in grad_honest:
        #     with open('norm_log_2/grad_honest.txt', 'a') as f:
        #         f.write(f'{grad}\n{torch.norm(grad)}\n\n')
        # with open('norm_log_2/grad_honest.txt', 'a') as f:
        #         f.write('************************')
                
                
        cache = self.get_flatten_parameters()
        # state_cache = self.model.state_dict()
        opt_cache = deepcopy(self.optimizer.state_dict())
        
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.shuffled_y)  # compute loss with shuffled labels
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        vic_gradient = self.get_flatten_parameters() - cache
        # flattened_gradients = torch.cat([param.grad.flatten() for param in self.model.parameters() if param.grad is not None])
        # vic_gradient = flattened_gradients.clone()
        
        # with open('norm_log_2/grad_vic.txt', 'a') as f:
        #     f.write(f'{vic_gradient}\n{torch.norm(vic_gradient)}\n\n')
        
        # self.model.load_state_dict(state_cache)  # 恢复模型状态
        self.load_parameters(cache)
        self.optimizer.load_state_dict(opt_cache)

        

        # 获得cover梯度
        candidate_cover_samples : list = self.data_reader.reserve_set # candidate cover set (len: 530)
        # 将candidate_cover_samples打乱
        # print(candidate_cover_samples)
        # rand_indices = torch.randperm(len(candidate_cover_samples))
        # candidate_cover_samples = candidate_cover_samples[rand_indices]
        # print('##################################')
        # print(candidate_cover_samples)
        
        

        cur_max_agrEvader_grad = None
        # 选取其中的300个作为本轮的cover set
        
        # TODO 在这里修改
        # 对每个cov中的样本单独计算梯度并保存
        # single_cover_gradient = torch.zeros(0).to(self.DEVICE)
        single_cover_gradient = []
        for sample in candidate_cover_samples:
            x, y = self.data_reader.get_batch(sample)
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            single_cover_gradient.append(self.get_flatten_parameters() - cache)
            # single_cover_gradient.append( torch.cat([param.grad.flatten() for param in self.model.parameters() if param.grad is not None]) )

            # self.model.load_state_dict(state_cache)  # 恢复模型状态
            self.load_parameters(cache)
            self.optimizer.load_state_dict(opt_cache)
            
        
        single_cover_gradient = torch.stack(single_cover_gradient)
        
        # for i, grad_cover in enumerate(single_cover_gradient):
        #     with open('norm_log_2/grad_cover.txt', 'a') as f:
        #         f.write(f'cov_{i}: {grad_cover}\n{torch.norm(grad_cover)}\n\n')
        # with open('norm_log_2/grad_cover.txt', 'a') as f:
        #      f.write('************************\n\n')   
        
        # print(f'single_cover_gradient={len(single_cover_gradient)}')
        # print(f'candidate_cover_samples={len(candidate_cover_samples)}')
        assert len(single_cover_gradient)==len(candidate_cover_samples)
        
        # 每一轮挑选一个最好的梯度
        actual_selected_index = []
        selected_grad_index = []
        selected_grad = []
        best_agr_grad = torch.zeros(0).to(self.DEVICE)
        best_g_cov = None
        
        gradient = None
        indices = None
        
        for j in range(5):
            
            # selected_grad_index = []
            cur_max_agrEvader_grad = torch.zeros(0).to(self.DEVICE)
            # best_agr_grad = torch.zeros(0).to(self.DEVICE)
            index = None
            gradient = None
            
            for i, g in enumerate(single_cover_gradient):
                if i in selected_grad_index:
                    continue
                # 组合cover和poison
                if len(selected_grad_index) == 0:
                    # gradient, indices = select_by_threshold(g*cover_factor+vic_gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
                    g_cov = g
                    gradient = g*cover_factor+vic_gradient
                else:
                    selected_grad_temp = torch.stack(selected_grad)
                    g_cov = ((torch.sum(selected_grad_temp, dim=0) + g)/(len(selected_grad)+1))
                    # gradient, indices = select_by_threshold(g_cov*cover_factor+vic_gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
                    gradient = g_cov*cover_factor+vic_gradient
                # limitation
                max_diff = 0.0
                for k in range(len(grad_honest)):
                    max_diff = max(torch.norm(gradient - grad_honest[k]), max_diff)

                if max_diff <= max_honest_diff and  torch.norm(gradient) > torch.norm(cur_max_agrEvader_grad):
                    cur_max_agrEvader_grad = gradient
                    index = i
                    
                    
            if cur_max_agrEvader_grad.numel() != 0:
                # print(f'*****************************')
                # print('find a satisfied gradient!')
                # print(f'*****************************')
                selected_grad_index.append(index)
                selected_grad.append(single_cover_gradient[index])
                if torch.norm(cur_max_agrEvader_grad) > torch.norm(best_agr_grad):
                    best_agr_grad = cur_max_agrEvader_grad 
                    actual_selected_index.append(index) 
                    best_g_cov = deepcopy(g_cov)
            # print(f'*****************************')
            # print(f'index={index}')
            # print(f'*****************************')
        # print(f'*****************************')
        # print(selected_grad_index)
        # print(f'*****************************')
        logger.info(f'selected grad index: {selected_grad_index}')
        logger.info(f'actual selected grad index: {actual_selected_index}')
                
            
        # print(f'selected_grad: {selected_grad}, len: {len(selected_grad)}')
        if best_agr_grad.numel() == 0:
            self.aggregator.collect(vic_gradient, indices)
            print('return vic gradient')
            return vic_gradient
        else:
            # optimize cover_factor
            # optimal_cover_factor = self.optimize_cover_factor(vic_gradient, best_g_cov, max_honest_diff, grad_honest)
            optimal_cover_factor = 1.0
            logger.info(f'cover factor is optimized from 0.5 to {optimal_cover_factor}')
            best_agr_grad = optimal_cover_factor * best_g_cov + vic_gradient
            self.aggregator.collect(best_agr_grad, indices)
            print('return agr gradient')
            return best_agr_grad

        '''
        for _ in range(5):
        # while True:
            rand_indices = torch.randperm(candidate_cover_samples.size(0))
            selected_indices = rand_indices[:320]
            cover_samples = candidate_cover_samples[selected_indices]
            assert len(cover_samples) == 320, "cover set size is not 320"

            i = 0
            while i * batch_size < len(cover_samples):
                batch_index = cover_samples[i * batch_size:(i + 1) * batch_size]
                x, y = self.data_reader.get_batch(batch_index)
                out = self.model(x)
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
            cover_gradient = self.get_flatten_parameters() - cache
        
            # 组合cover和poison
            if RESERVED_SAMPLE != 0:
                gradient, indices = select_by_threshold(cover_gradient*cover_factor+gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD) # computed the malicious gradient
            else:
                gradient, indices = select_by_threshold(gradient,GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)

            # limitation
            max_diff = 0.0
            for k in range(len(grad_honest)):
                max_diff = max(torch.norm(gradient  - grad_honest[k]), max_diff)

            if max_diff <= max_honest_diff:
                print("find a satisfied gradient!")
                # break
                if cur_max_agrEvader_grad == None:
                    cur_max_agrEvader_grad = gradient
                elif torch.norm(gradient) > torch.norm(cur_max_agrEvader_grad):
                    cur_max_agrEvader_grad = gradient

        # 判断使用新生成的还是使用历史最佳的
        if cur_max_agrEvader_grad == None:
            self.aggregator.collect(gradient, indices)
            return gradient
        else :
            self.aggregator.collect(cur_max_agrEvader_grad, indices)
            return cur_max_agrEvader_grad
        '''

        '''
        if cur_max_agrEvader_grad == None and self.best_AGREvader_grad == None:
            # raise ValueError("attack not find AGREvader gradient!")
            self.aggregator.collect(gradient, indices)
            return gradient

        if self.best_AGREvader_grad == None:
            print("111111111111111")
            self.best_AGREvader_grad = cur_max_agrEvader_grad
        elif cur_max_agrEvader_grad == None:
            print("=> keep using history best AGREvader gradient")
        elif torch.norm(cur_max_agrEvader_grad) > torch.norm(self.best_AGREvader_grad):
            print(torch.norm(cur_max_agrEvader_grad), torch.norm(self.best_AGREvader_grad))
            print("=> attacker find better AGREvader gradient")
            # raise ValueError("attacker find better AGREvader gradient")
            self.best_AGREvader_grad = cur_max_agrEvader_grad
            

        self.aggregator.collect(self.best_AGREvader_grad, indices)
        return self.best_AGREvader_grad*/
        '''
                
    def blackbox_attack_unit(self,num_malicious=1, cover_factor = 0,batch_size = BATCH_SIZE, grad_honest = None, logger=None):
        """
        Optimized shuffle label attack
        :param cover_factor: Cover factor of the gradient of cover samples
        :param batch_size: The size of the batch
        :return: The malicious gradient covered by gradient of cover samples for current round
        """
        # assert len(grad_honest) == NUMBER_OF_PARTICIPANTS, "honest gradient have not fully collected!"
        # 获取max_honest_diff
        max_honest_diff = 0.0
        for i in range(len(grad_honest)):
            for j in range(i+1, len(grad_honest)):
                grad_i = grad_honest[i] / torch.norm(grad_honest[i], p=2)
                grad_j = grad_honest[j] / torch.norm(grad_honest[j], p=2)
                max_honest_diff = max(max_honest_diff, torch.norm(grad_i - grad_j))
                
        # 获得poison梯度
        cache = self.get_flatten_parameters()
        op_cache = deepcopy(self.optimizer.state_dict())
        
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.shuffled_y) # compute loss with shuffled labels
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        vic_gradient = self.get_flatten_parameters() - cache
        
        self.load_parameters(cache)
        self.optimizer.load_state_dict(op_cache)

        # 获得cover梯度
        candidate_cover_samples = self.data_reader.reserve_set # candidate cover set (len: 530)
        cur_max_agrEvader_grad = None
        # 选取其中的300个作为本轮的cover set
        
        single_cover_gradient = []
        for sample in candidate_cover_samples:
            x, y = self.data_reader.get_batch(sample)
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            single_cover_gradient.append(self.get_flatten_parameters() - cache)
            
            self.load_parameters(cache)
            self.optimizer.load_state_dict(op_cache)
        
        single_cover_gradient = torch.stack(single_cover_gradient)
        
        # print(f'single_cover_gradient={len(single_cover_gradient)}')
        # print(f'candidate_cover_samples={len(candidate_cover_samples)}')
        assert len(single_cover_gradient)==len(candidate_cover_samples)
        
        # 每一轮挑选一个最好的梯度
        actual_selected_index = []
        selected_grad_index = []
        selected_grad = []
        best_agr_grad = torch.zeros(0).to(self.DEVICE)
        best_g_cov = None
        
        gradient = None
        indices = None
        
        for j in range(5):
            
            # selected_grad_index = []
            cur_max_agrEvader_grad = torch.zeros(0).to(self.DEVICE)
            # best_agr_grad = torch.zeros(0).to(self.DEVICE)
            index = None
            gradient = None
            
            for i, g in enumerate(single_cover_gradient):
                # # 将每一个cover梯度写入文件
                if i in selected_grad_index:
                    continue
                # 组合cover和poison
                if len(selected_grad_index) == 0:
                    # gradient, indices = select_by_threshold(g*cover_factor+vic_gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
                    g_cov = g
                    gradient = g*cover_factor+vic_gradient 
                else:
                    selected_grad_temp = torch.stack(selected_grad)
                    g_cov = ((torch.sum(selected_grad_temp, dim=0) + g)/(len(selected_grad)+1))
                    # gradient, indices = select_by_threshold(g_cov*cover_factor+vic_gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
                    gradient = g_cov*cover_factor+vic_gradient
                # limitation
                max_diff = 0.0
                for k in range(len(grad_honest)):
                     max_diff = max(torch.norm(gradient / torch.norm(gradient, p=2) - grad_honest[k] / torch.norm(grad_honest[k], p=2)), max_diff)
                # print(f'max_honest_diff={max_honest_diff}, max_diff={max_diff}')
                # 将max_diff 写入文件
                # with open('max_diff.txt', 'a') as f:
                #     f.write(f'{j}_{i}: {max_diff} *** {max_honest_diff}\n')
                    
                # with open('norm.txt', 'a') as f:
                #     f.write(f'{j}_{i}: {torch.norm(gradient)} *** {torch.norm(cur_max_agrEvader_grad)}\n')

                if max_diff <= max_honest_diff and  torch.norm(gradient) > torch.norm(cur_max_agrEvader_grad):

                    cur_max_agrEvader_grad = gradient
                    index = i
                    
                    
            if cur_max_agrEvader_grad.numel() != 0:
                # print(f'*****************************')
                # print('find a satisfied gradient!')
                # print(f'*****************************')
                selected_grad_index.append(index)
                selected_grad.append(single_cover_gradient[index])
                if torch.norm(cur_max_agrEvader_grad) > torch.norm(best_agr_grad):
                    best_agr_grad = cur_max_agrEvader_grad 
                    actual_selected_index.append(index) 
                    best_g_cov = deepcopy(g_cov)

            # print(f'*****************************')
            # print(f'index={index}')
            # print(f'*****************************')
        # print(f'*****************************')
        # print(selected_grad_index)
        # print(f'*****************************')
        logger.info(f'selected grad index: {selected_grad_index}')
        logger.info(f'actual selected grad index: {actual_selected_index}')
            
        # print(f'selected_grad: {selected_grad}, len: {len(selected_grad)}')
        if best_agr_grad.numel() == 0:
            self.aggregator.collect(vic_gradient, indices, num_malicious)
            print('return vic gradient')
            return vic_gradient
        else:
            # optimize cover_factor
            # optimal_cover_factor = self.optimize_cover_factor(vic_gradient, best_g_cov, max_honest_diff, grad_honest)
            # logger.info(f'cover factor is optimized from 0.5 to {optimal_cover_factor}')
            # best_agr_grad = optimal_cover_factor * best_g_cov + vic_gradient
            self.aggregator.collect(best_agr_grad, indices, num_malicious)
            print('return agr gradient')
            return best_agr_grad

        
        '''
        for _ in range(5):
        # while True:
            rand_indices = torch.randperm(candidate_cover_samples.size(0))
            selected_indices = rand_indices[:320]
            cover_samples = candidate_cover_samples[selected_indices]
            assert len(cover_samples) == 320, "cover set size is not 300"

            i = 0
            while i * batch_size < len(cover_samples):
                batch_index = cover_samples[i * batch_size:(i + 1) * batch_size]
                x, y = self.data_reader.get_batch(batch_index)
                out = self.model(x)
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
            cover_gradient = self.get_flatten_parameters() - cache
        
            # 组合cover和poison
            if RESERVED_SAMPLE != 0:
                gradient, indices = select_by_threshold(cover_gradient*cover_factor+gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD) # computed the malicious gradient
            else:
                gradient, indices = select_by_threshold(gradient,GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)

            # limitation
            max_diff = 0.0
            for k in range(len(grad_honest)):
                max_diff = max(torch.norm(gradient / torch.norm(gradient, p=2) - grad_honest[k] / torch.norm(grad_honest[k], p=2)), max_diff)

            if max_diff <= max_honest_diff:
                # break
                if cur_max_agrEvader_grad == None:
                    cur_max_agrEvader_grad = gradient
                elif torch.norm(gradient) > torch.norm(cur_max_agrEvader_grad):
                    # print("find a better gradient!")s
                    cur_max_agrEvader_grad = gradient

        # 判断使用新生成的还是使用历史最佳的
        if cur_max_agrEvader_grad == None:
            self.aggregator.collect(gradient, indices)
            # self.aggregator.collect(gradient /torch.norm(gradient), indices)
            return gradient
        else :
            self.aggregator.collect(cur_max_agrEvader_grad, indices)
            # self.aggregator.collect(cur_max_agrEvader_grad / torch.norm(cur_max_agrEvader_grad), indices)
            return cur_max_agrEvader_grad
        '''
        
    def blackbox_attack_ascent(self):
        """
        do gradient ascent on the victim samples
        """
        cache = self.get_flatten_parameters()
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        # add minus to the gradient 
        self.aggregator.collect(-gradient)
        return gradient
        
        
    def get_angle(self, gradA, gradB):
        if isinstance(gradA, torch.Tensor) and isinstance(gradB, torch.Tensor):
            # 将 gradA 和 gradB 展平为 1D 向量
            dot_product = torch.dot(gradA.view(-1), gradB.view(-1))
            norm_A = torch.norm(gradA, p=2)
            norm_B = torch.norm(gradB, p=2)

            # 计算余弦相似度
            cos_theta = dot_product / (norm_A * norm_B)
            # 使用 clip 限制 cos_theta 的范围，防止浮点数误差
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            
            # 计算夹角（弧度），并转换为角度
            theta = torch.acos(cos_theta)
            theta_degrees = torch.rad2deg(theta)

        elif isinstance(gradA, np.ndarray) and isinstance(gradB, np.ndarray):
            # 计算 NumPy 向量的点积和范数
            dot_product = np.dot(gradA.flatten(), gradB.flatten())
            norm_A = np.linalg.norm(gradA)
            norm_B = np.linalg.norm(gradB)
            
            # 计算余弦相似度
            cos_theta = dot_product / (norm_A * norm_B)
            # 使用 clip 限制 cos_theta 的范围，防止浮点数误差
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            
            # 计算夹角（弧度），并转换为角度
            theta = np.arccos(cos_theta)
            theta_degrees = np.degrees(theta)

        else:
            raise ValueError('Input should be either torch.Tensor or numpy.ndarray')
        
        return theta_degrees
    
    def get_cos(self, gradA: torch.Tensor, gradB: torch.Tensor):
        # dot_product = torch.dot(gradA, gradB)
        dot_product = torch.dot(gradA.view(-1), gradB.view(-1))
        norm_A = torch.norm(gradA, p=2)
        norm_B = torch.norm(gradB, p=2)
        cos_theta = dot_product / (norm_A * norm_B)

        return cos_theta

    def optimize_cover_factor(self, vic_gradient, cover_gradient, honest_max_diff, honest_grad_lst):
            vic_gradient = vic_gradient.detach().cpu().numpy()
            cover_gradient = cover_gradient.detach().cpu().numpy()
            honest_grad_lst = [honest_grad.detach().cpu().numpy() for honest_grad in honest_grad_lst]
            honest_max_diff = honest_max_diff.cpu().numpy()
            # print(f'{vic_gradient.device}, {cover_gradient.device}, {honest_grad_lst[0].device}')
            assert isinstance(vic_gradient, np.ndarray)
            assert isinstance(cover_gradient, np.ndarray)
            assert isinstance(honest_grad_lst[0], np.ndarray)

            def objective(cover_factor, g, vic_gradient):
                gradient = g * cover_factor + vic_gradient
                return -np.linalg.norm(gradient)  # 最大化 norm，因此最小化其负值

            def constraint(cover_factor, g, vic_gradient, honest_max_diff, honest_grad_lst):
                gradient = g * cover_factor + vic_gradient
                agr_honest_max_diff = 0.0
                for honest_grad in honest_grad_lst:
                    agr_honest_max_diff = max(self.get_angle(gradient, honest_grad), agr_honest_max_diff)
                return honest_max_diff - agr_honest_max_diff  # 确保 agr_honest_max_diff < honest_max_diff

            # 优化 cover_factor，添加约束
            bounds = [(0, 1)]  # 限制 cover_factor 在 [0, 1] 之间
            cons = {'type': 'ineq', 'fun': constraint, 'args': (cover_gradient, vic_gradient, honest_max_diff, honest_grad_lst)}

            result = minimize(objective, x0=[0.5], args=(cover_gradient, vic_gradient), bounds=bounds, constraints=cons)

            optimal_cover_factor = result.x[0]
            return optimal_cover_factor
    
    def blackbox_attack_angle(self,num_malicious=1, cover_factor = 0,batch_size = BATCH_SIZE, grad_honest = None, logger=None):
        """
        Optimized shuffle label attack
        :param cover_factor: Cover factor of the gradient of cover samples
        :param batch_size: The size of the batch
        :return: The malicious gradient covered by gradient of cover samples for current round
        """
        # assert len(grad_honest) == NUMBER_OF_PARTICIPANTS, "honest gradient have not fully collected!"
        # 获取max_honest_diff
        max_honest_diff = 0.0
        #! 只获取8个 
        # grad_honest = grad_honest[:8]
        # print(len(grad_honest))
        for i in range(len(grad_honest)):
            for j in range(i+1, len(grad_honest)):
                theta_degrees = self.get_angle(grad_honest[i], grad_honest[j])
                max_honest_diff = max(max_honest_diff, theta_degrees)
                
        cache = self.get_flatten_parameters()
        # state_cache = self.model.state_dict()
        opt_cache = deepcopy(self.optimizer.state_dict())
        
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.shuffled_y)  # compute loss with shuffled labels
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        vic_gradient = self.get_flatten_parameters() - cache
        # flattened_gradients = torch.cat([param.grad.flatten() for param in self.model.parameters() if param.grad is not None])
        # vic_gradient = flattened_gradients.clone()
        
        # with open('angle_log_2/grad_vic.txt', 'a') as f:
        #     f.write(f'{vic_gradient}\n{torch.norm(vic_gradient)}\n\n')
        
        # self.model.load_state_dict(state_cache)  # 恢复模型状态
        self.load_parameters(cache)
        self.optimizer.load_state_dict(opt_cache) 


        # 获得cover梯度
        candidate_cover_samples = self.data_reader.reserve_set # candidate cover set (len: 530)
        cur_max_agrEvader_grad = None
        
        single_cover_gradient = []
        for sample in candidate_cover_samples:
            x, y = self.data_reader.get_batch(sample)
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            single_cover_gradient.append(self.get_flatten_parameters() - cache)
            # single_cover_gradient.append( torch.cat([param.grad.flatten() for param in self.model.parameters() if param.grad is not None]) )

            # self.model.load_state_dict(state_cache)  # 恢复模型状态
            self.load_parameters(cache)
            self.optimizer.load_state_dict(opt_cache)
            
        single_cover_gradient = torch.stack(single_cover_gradient)
        
        # for i, grad_cover in enumerate(single_cover_gradient):
        #     with open('angle_log_2/grad_cover.txt', 'a') as f:
        #         f.write(f'cov_{i}: {grad_cover}\n{torch.norm(grad_cover)}\n\n')
        # with open('angle_log_2/grad_cover.txt', 'a') as f:
        #      f.write('************************\n\n')   
        
        # print(f'single_cover_gradient={len(single_cover_gradient)}')
        # print(f'candidate_cover_samples={len(candidate_cover_samples)}')
        assert len(single_cover_gradient)==len(candidate_cover_samples)
        
        
        # # 计算每个cover_gradient和vic_gradient的夹角并写入文件
        # for i, g in enumerate(single_cover_gradient):
        #     theta_degrees = self.get_angle(g, vic_gradient)
        #     with open('./angle_log_2/cov_vic_angle.txt', 'a') as f:
        #         f.write(f'{i}: {theta_degrees}\n')
        #         
        #     # 计算每个cover_gradient和vic_gradient的norm并写入文件
        #     with open('./angle_log_2/cov_vic_norm.txt', 'a') as f:
        #         f.write(f'{i}: cov: {torch.norm(g)} *** vic: {torch.norm(vic_gradient)}\n')
        #         
        #     # 计算vic_gradient和honest_gradient的夹角并写入文件
        #     with open('./angle_log_2/vic_honest_angle.txt', 'a') as f:
        #         for k in range(len(grad_honest)):
        #             theta_degrees = self.get_angle(vic_gradient, grad_honest[k])
        #             f.write(f'{i}_{k}: {theta_degrees}\n')
        #     
        # # 计算cover_gradient和honest_gradient的夹角并写入文件    
        # with open('./angle_log_2/cov_honest_angle.txt', 'a') as f:
        #     for k in range(len(grad_honest)):
        #         theta_degrees = self.get_angle(g, grad_honest[k])
        #         f.write(f'{i}_{k}: {theta_degrees}\n')
                    
        
        # 每一轮挑选一个最好的梯度
        actual_selected_index = []
        selected_grad_index = []
        selected_grad = []
        best_agr_grad = torch.zeros(0).to(self.DEVICE)
        best_g_cov = None
        
        gradient = None
        indices = None

        min_cover_honest_diff = 360.0
        for cover_grad in single_cover_gradient:
            max_cover_honest_diff = 0.0
            for honest_grad in grad_honest:
                max_cover_honest_diff = max(max_cover_honest_diff, self.get_angle(cover_grad + vic_gradient, honest_grad))
            min_cover_honest_diff = min(min_cover_honest_diff, max_cover_honest_diff)

        
        for j in range(5):
            
            # selected_grad_index = []
            cur_max_agrEvader_grad = torch.zeros(0).to(self.DEVICE)
            # best_agr_grad = torch.zeros(0).to(self.DEVICE)
            index = None
            gradient = None
            g_cov = None
            
            for i, g in enumerate(single_cover_gradient):
                if i in selected_grad_index:
                    continue
                # 组合cover和poison
                if len(selected_grad_index) == 0:
                    # gradient, indices = select_by_threshold(g*cover_factor+vic_gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
                    g_cov = g
                    gradient = g_cov * cover_factor + vic_gradient
                else:
                    selected_grad_temp = torch.stack(selected_grad)
                    g_cov = ((torch.sum(selected_grad_temp, dim=0) + g)/(len(selected_grad)+1))
                    # gradient, indices = select_by_threshold(g_cov*cover_factor+vic_gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
                    gradient = g_cov * cover_factor + vic_gradient
                # limitation
                # gradient = gradient + grad_honest[0]
                
                max_diff = 0.0
                for k in range(len(grad_honest)):
                    theta = self.get_angle(gradient, grad_honest[k])
                    max_diff = max(theta, max_diff)
                if max_diff <= max_honest_diff and  torch.norm(gradient) > torch.norm(cur_max_agrEvader_grad):

                    cur_max_agrEvader_grad = gradient
                    index = i
                    
            if cur_max_agrEvader_grad.numel() != 0:
                # print(f'*****************************')
                # print('find a satisfied gradient!')
                # print(f'*****************************')
                selected_grad_index.append(index)
                selected_grad.append(single_cover_gradient[index])
                if torch.norm(cur_max_agrEvader_grad) > torch.norm(best_agr_grad):
                    best_agr_grad = cur_max_agrEvader_grad 
                    actual_selected_index.append(index) 
                    best_g_cov = deepcopy(g_cov)
                    # print(f'g cover type: {type(g_cov)}, best g cover type: {type(best_g_cov)}')
            # print(f'*****************************')
            # print(f'index={index}')
            # print(f'*****************************')
        # print(f'*****************************')
        # print(selected_grad_index)
        # print(f'*****************************')
        logger.info(f'max_cover_honest_diff: {min_cover_honest_diff}, max_honest_diff: {max_honest_diff}')
        logger.info(f'selected grad index: {selected_grad_index}')
        logger.info(f'actual selected grad index: {actual_selected_index}')
            
        if best_agr_grad.numel() == 0:
            self.aggregator.collect(vic_gradient, indices, num_malicious)
            print('return vic gradient')
            return vic_gradient
        else:
            # optimize cover_factor
            # optimal_cover_factor = self.optimize_cover_factor(vic_gradient, best_g_cov, max_honest_diff, grad_honest)
            # logger.info(f'cover factor is optimized from 0.5 to {optimal_cover_factor}')
            # best_agr_grad = optimal_cover_factor * best_g_cov + vic_gradient
            self.aggregator.collect(best_agr_grad, indices, num_malicious)
            print('return agr gradient')
            return best_agr_grad
        
        '''
        history_max_diff = 0.0
        for _ in range(try_times):
            rand_indices = torch.randperm(candidate_cover_samples.size(0))
            selected_indices = rand_indices[:320]
            cover_samples = candidate_cover_samples[selected_indices]
            assert len(cover_samples) == 320, "cover set size is not 300"

            i = 0
            while i * batch_size < len(cover_samples):
                batch_index = cover_samples[i * batch_size:(i + 1) * batch_size]
                x, y = self.data_reader.get_batch(batch_index)
                out = self.model(x)
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
            cover_gradient = self.get_flatten_parameters() - cache
        
            # 组合cover和poison
            if RESERVED_SAMPLE != 0:
                gradient, indices = select_by_threshold(cover_gradient*cover_factor+gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD) # computed the malicious gradient
            else:
                gradient, indices = select_by_threshold(gradient,GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)

            # limitation
            max_diff = 0.0
            for k in range(len(grad_honest)):
                theta = self.get_angle(gradient, grad_honest[k])
                max_diff = max(theta, max_diff)

            if max_diff <= max_honest_angle_diff:
                if cur_max_agrEvader_grad == None:
                    history_max_diff = max_diff
                    cur_max_agrEvader_grad = gradient
                # elif torch.norm(gradient) > torch.norm(cur_max_agrEvader_grad):
                elif max_diff > history_max_diff:
                    history_max_diff = max_diff
                    print("find a better gradient!")
                    cur_max_agrEvader_grad = gradient

        # 判断使用新生成的还是使用历史最佳的
        if cur_max_agrEvader_grad == None:
            # self.aggregator.collect(gradient / torch.norm(gradient), indices)
            self.aggregator.collect(gradient, indices)
            return gradient
        else :
            # self.aggregator.collect(cur_max_agrEvader_grad / torch.norm(cur_max_agrEvader_grad), indices)
            self.aggregator.collect(cur_max_agrEvader_grad, indices)
            return cur_max_agrEvader_grad
        '''
    
    def blackbox_attack_adaptive(self, num_malicious=1, cover_factor=0, batch_size=BATCH_SIZE, grad_honest=None, logger=None):
        # 得到victim gradient
        cache = self.get_flatten_parameters()
        opt_cache = deepcopy(self.optimizer.state_dict())

        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.shuffled_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        vic_gradient = self.get_flatten_parameters() - cache

        self.load_parameters(cache)
        self.optimizer.load_state_dict(opt_cache)

        # 迭代进行angle-trim判断，保证我们的攻击梯度不被筛选掉
        mean_grad_honest = torch.mean(torch.stack(grad_honest), dim=0)

        grad_honest.append(vic_gradient)
        num_gradients = len(grad_honest)
        angles = torch.zeros(num_gradients, num_gradients)

        # 计算每一对梯度之间的角度
        for i in range(num_gradients):
            for j in range(i + 1, num_gradients):
                g1 = grad_honest[i]
                g2 = grad_honest[j]

                cos_theta = F.cosine_similarity(g1, g2, dim=0).item()
                cos_theta_clamped = max(-1.0, min(1.0, cos_theta))
                theta = torch.acos(torch.tensor(cos_theta_clamped))
                angles[i, j] = theta
                angles[j, i] = theta  # 对称矩阵

        # 计算每个梯度的平均角度
        mean_angles = angles.mean(dim=1)

        # 判断攻击梯度是否被去除
        attack_gradient = vic_gradient
        while True:
            # 计算攻击梯度的平均角度
            attack_angle = mean_angles[-1]  # 这里是攻击梯度的平均角度

            # 排序所有梯度的平均角度
            sorted_indices = torch.argsort(mean_angles)
            sorted_mean_angles = mean_angles[sorted_indices]

            # 找出去除位置的索引
            selected_indices = sorted_indices[:- 2 * num_malicious]  # 根据原始angle_trim逻辑，去除最大角度的部分

            # 如果攻击梯度的平均角度在去除的位置之外，则说明攻击成功
            if attack_angle < sorted_mean_angles[- 2 * num_malicious]:
                print('1111')
                break
            else:
                print('2222')
                # 继续与与其角度差最大的良性梯度相加
                max_diff_index = torch.argmax(angles[-1, :-1])
                print(max_diff_index)
                # combined_gradient = (attack_gradient + grad_honest[max_diff_index])
                # combined_gradient = torch.mean(attack_gradient, grad_honest[max_diff_index])
                combined_gradient = (attack_gradient + grad_honest[max_diff_index]) / 2  # 取平均值
                # combined_gradient = (attack_gradient + mean_grad_honest) / 2  # 取平均值

                print(attack_gradient)
                print(grad_honest[max_diff_index])
                # print(mean_grad_honest)
                print(combined_gradient)

                # 更新攻击梯度并重新计算平均角度
                attack_gradient = combined_gradient
                
                grad_honest[-1] = attack_gradient
                print(grad_honest[-1])
                # mean_angles[-1] = angles[grad_honest.index(attack_gradient), :].mean()  # 更新攻击梯度的平均角度
                for i in range(num_gradients - 1):
                    g1 = grad_honest[-1]
                    g2 = grad_honest[i]

                    cos_theta = F.cosine_similarity(g1, g2, dim=0).item()
                    cos_theta_clamped = max(-1.0, min(1.0, cos_theta))
                    theta = torch.acos(torch.tensor(cos_theta_clamped))
                    angles[-1, i] = theta
                    angles[i, -1] = theta  # 对称矩阵
                mean_angles = angles.mean(dim=1)

        # 返回最终的攻击梯度
        return attack_gradient



    def blackbox_attack_angle_new(self,num_malicious=1, cover_factor = 0,batch_size = BATCH_SIZE, grad_honest = None, logger=None):
        """
        Optimized shuffle label attack
        :param cover_factor: Cover factor of the gradient of cover samples
        :param batch_size: The size of the batch
        :return: The malicious gradient covered by gradient of cover samples for current round
        """
        # 获取max_honest_diff
        max_honest_diff = 0.0
        for i in range(len(grad_honest)):
            for j in range(i+1, len(grad_honest)):
                theta_degrees = self.get_angle(grad_honest[i], grad_honest[j])
                max_honest_diff = max(max_honest_diff, theta_degrees)
                
        cache = self.get_flatten_parameters()
        opt_cache = deepcopy(self.optimizer.state_dict())
        
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.shuffled_y)  # compute loss with shuffled labels
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        vic_gradient = self.get_flatten_parameters() - cache
        
        self.load_parameters(cache)
        self.optimizer.load_state_dict(opt_cache) 

        # 获得cover梯度
        candidate_cover_samples = self.data_reader.reserve_set # candidate cover set (len: 530)
        cur_max_agrEvader_grad = None
        
        single_cover_gradient = []
        for sample in candidate_cover_samples:
            x, y = self.data_reader.get_batch(sample)
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            single_cover_gradient.append(self.get_flatten_parameters() - cache)
            # single_cover_gradient.append( torch.cat([param.grad.flatten() for param in self.model.parameters() if param.grad is not None]) )

            # self.model.load_state_dict(state_cache)  # 恢复模型状态
            self.load_parameters(cache)
            self.optimizer.load_state_dict(opt_cache)
            
        single_cover_gradient = torch.stack(single_cover_gradient)
        

        assert len(single_cover_gradient)==len(candidate_cover_samples)        
        
        # 每一轮挑选一个最好的梯度
        actual_selected_index = []
        selected_grad_index = []
        selected_grad = []
        best_agr_grad = torch.zeros(0).to(self.DEVICE)
        
        gradient = None
        indices = None

        min_cover_honest_diff = 360.0
        for cover_grad in single_cover_gradient:
            max_cover_honest_diff = 0.0
            for honest_grad in grad_honest:
                max_cover_honest_diff = max(max_cover_honest_diff, self.get_angle(cover_grad + vic_gradient, honest_grad))
            min_cover_honest_diff = min(min_cover_honest_diff, max_cover_honest_diff)

        cnt = 0
        while len(selected_grad_index) < 5 and cnt < 50:
            cur_max_agrEvader_grad = torch.zeros(0).to(self.DEVICE)
            index = None
            gradient = None
            g_cov = None
            
            history_max_diff = 0.0
            for i, g in enumerate(single_cover_gradient):
                if i in selected_grad_index:
                    continue
                # 组合cover和poison
                if len(selected_grad_index) == 0:
                    g_cov = g
                    gradient = g_cov * cover_factor + vic_gradient
                else:
                    selected_grad_temp = torch.stack(selected_grad)
                    g_cov = ((torch.sum(selected_grad_temp, dim=0) + g)/(len(selected_grad)+1))
                    gradient = g_cov * cover_factor + vic_gradient
                
                max_diff = 0.0
                # 计算 maliciuous gradient 跟 honest gradiets的最大夹角
                for k in range(len(grad_honest)):
                    theta = self.get_angle(gradient, grad_honest[k])
                    max_diff = max(theta, max_diff)
                # 找到当前样本中能够满足约束同时能够使得角度差最大的样本
                if max_diff <= max_honest_diff and max_diff > history_max_diff:
                    cur_max_agrEvader_grad = gradient
                    index = i
                    history_max_diff = max_diff
                    
            if index is not None:
                selected_grad_index.append(index)
                selected_grad.append(single_cover_gradient[index])
                best_agr_grad = cur_max_agrEvader_grad
            
            cnt += 1

        logger.info(f'max_cover_honest_diff: {min_cover_honest_diff}, max_honest_diff: {max_honest_diff}')
        logger.info(f'selected grad index: {selected_grad_index}')
            
        if best_agr_grad.numel() == 0:
            self.aggregator.collect(vic_gradient, indices, num_malicious)
            print('return vic gradient')
            return vic_gradient
        else:
            # optimize cover_factor
            # optimal_cover_factor = self.optimize_cover_factor(vic_gradient, best_g_cov, max_honest_diff, grad_honest)
            # logger.info(f'cover factor is optimized from 0.5 to {optimal_cover_factor}')
            # best_agr_grad = optimal_cover_factor * best_g_cov + vic_gradient
            self.aggregator.collect(best_agr_grad, indices, num_malicious)
            print('return agr gradient')
            return best_agr_grad
        
         
    def blackbox_attack_angle_random(self,num_malicious=1, cover_factor = 0,batch_size = BATCH_SIZE, grad_honest = None, logger=None):
        """
        Optimized shuffle label attack
        :param cover_factor: Cover factor of the gradient of cover samples
        :param batch_size: The size of the batch
        :return: The malicious gradient covered by gradient of cover samples for current round
        """
        # assert len(grad_honest) == NUMBER_OF_PARTICIPANTS, "honest gradient have not fully collected!"
        # 获取max_honest_diff
        max_honest_diff = 0.0
        for i in range(len(grad_honest)):
            for j in range(i+1, len(grad_honest)):
                theta_degrees = self.get_angle(grad_honest[i], grad_honest[j])
                max_honest_diff = max(max_honest_diff, theta_degrees)
                
        cache = self.get_flatten_parameters()
        # state_cache = self.model.state_dict()
        opt_cache = deepcopy(self.optimizer.state_dict())
        
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.shuffled_y)  # compute loss with shuffled labels
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        vic_gradient = self.get_flatten_parameters() - cache
        # flattened_gradients = torch.cat([param.grad.flatten() for param in self.model.parameters() if param.grad is not None])
        # vic_gradient = flattened_gradients.clone()
        
        # with open('angle_log_2/grad_vic.txt', 'a') as f:
        #     f.write(f'{vic_gradient}\n{torch.norm(vic_gradient)}\n\n')
        
        # self.model.load_state_dict(state_cache)  # 恢复模型状态
        self.load_parameters(cache)
        self.optimizer.load_state_dict(opt_cache) 


        # 获得cover梯度
        candidate_cover_samples = self.data_reader.reserve_set # candidate cover set (len: 530)
        cur_max_agrEvader_grad = None
        
        single_cover_gradient = []
        for sample in candidate_cover_samples:
            x, y = self.data_reader.get_batch(sample)
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            single_cover_gradient.append(self.get_flatten_parameters() - cache)
            # single_cover_gradient.append( torch.cat([param.grad.flatten() for param in self.model.parameters() if param.grad is not None]) )

            # self.model.load_state_dict(state_cache)  # 恢复模型状态
            self.load_parameters(cache)
            self.optimizer.load_state_dict(opt_cache)
            
        single_cover_gradient = torch.stack(single_cover_gradient)

        assert len(single_cover_gradient)==len(candidate_cover_samples)
        
        # 每一轮挑选一个最好的梯度
        actual_selected_index = []
        selected_grad_index = []
        selected_grad = []
        best_agr_grad = torch.zeros(0).to(self.DEVICE)
        best_g_cov = None
        
        gradient = None
        indices = None
        
        
        # 随机挑选30个样本
        selected_grad_index = random.sample(range(len(candidate_cover_samples)), 30)
        logger.info(f'selected grad index: {selected_grad_index}')
        selected_grad = [single_cover_gradient[i] for i in selected_grad_index]
        
        g_cov = torch.mean(torch.stack(selected_grad), dim=0)
        gradient = g_cov * cover_factor + vic_gradient
        
        max_diff = 0.0
        for k in range(len(grad_honest)):
            theta = self.get_angle(gradient, grad_honest[k])
            max_diff = max(theta, max_diff)
        if max_diff <= max_honest_diff:
            logger.info(f'return agr gradient')
            return gradient
        else:
            logger.info(f'return cov gradient')
            return g_cov            

        for j in range(5):
            
            # selected_grad_index = []
            cur_max_agrEvader_grad = torch.zeros(0).to(self.DEVICE)
            # best_agr_grad = torch.zeros(0).to(self.DEVICE)
            index = None
            gradient = None
            g_cov = None
            
            for i, g in enumerate(single_cover_gradient):
                if i in selected_grad_index:
                    continue
                # 组合cover和poison
                if len(selected_grad_index) == 0:
                    # gradient, indices = select_by_threshold(g*cover_factor+vic_gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
                    g_cov = g
                    gradient = g_cov * cover_factor + vic_gradient
                else:
                    selected_grad_temp = torch.stack(selected_grad)
                    g_cov = ((torch.sum(selected_grad_temp, dim=0) + g)/(len(selected_grad)+1))
                    # gradient, indices = select_by_threshold(g_cov*cover_factor+vic_gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
                    gradient = g_cov * cover_factor + vic_gradient
                # limitation
                # gradient = gradient + grad_honest[0]
                
                max_diff = 0.0
                for k in range(len(grad_honest)):
                    theta = self.get_angle(gradient, grad_honest[k])
                    max_diff = max(theta, max_diff)
                if max_diff <= max_honest_diff and  torch.norm(gradient) > torch.norm(cur_max_agrEvader_grad):
                    cur_max_agrEvader_grad = gradient
                    index = i
                    
            if cur_max_agrEvader_grad.numel() != 0:
                # print(f'*****************************')
                # print('find a satisfied gradient!')
                # print(f'*****************************')
                selected_grad_index.append(index)
                selected_grad.append(single_cover_gradient[index])
                if torch.norm(cur_max_agrEvader_grad) > torch.norm(best_agr_grad):
                    best_agr_grad = cur_max_agrEvader_grad 
                    actual_selected_index.append(index) 
                    best_g_cov = deepcopy(g_cov)
                    
                    
        

        logger.info(f'selected grad index: {selected_grad_index}')
        logger.info(f'actual selected grad index: {actual_selected_index}')
            
        if best_agr_grad.numel() == 0:
            self.aggregator.collect(vic_gradient, indices, num_malicious)
            print('return vic gradient')
            return vic_gradient
        else:
            # optimize cover_factor
            # optimal_cover_factor = self.optimize_cover_factor(vic_gradient, best_g_cov, max_honest_diff, grad_honest)
            # logger.info(f'cover factor is optimized from 0.5 to {optimal_cover_factor}')
            # best_agr_grad = optimal_cover_factor * best_g_cov + vic_gradient
            self.aggregator.collect(best_agr_grad, indices, num_malicious)
            print('return agr gradient')
            return best_agr_grad
        
    
        
    def blackbox_attack_cos(self,num_malicious=1, cover_factor = 0,batch_size = BATCH_SIZE, grad_honest = None): 
        """
        Optimized shuffle label attack
        :param cover_factor: Cover factor of the gradient of cover samples
        :param batch_size: The size of the batch
        :return: The malicious gradient covered by gradient of cover samples for current round
        """
        # assert len(grad_honest) == NUMBER_OF_PARTICIPANTS, "honest gradient have not fully collected!"
        # 获取max_honest_diff
        min_honest_cos_diff = 0.0
        for i in range(len(grad_honest)):
            for j in range(i+1, len(grad_honest)):
                theta = self.get_cos(grad_honest[i], grad_honest[j])
                min_honest_cos_diff = min(min_honest_cos_diff, theta)
        # 获得poison梯度
        cache = self.get_flatten_parameters()
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.shuffled_y) # compute loss with shuffled labels
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        self.load_parameters(cache)

        # 获得cover梯度
        candidate_cover_samples = self.data_reader.reserve_set # candidate cover set (len: 530)
        cur_max_agrEvader_grad = None
        # 选取其中的300个作为本轮的cover set
        history_min_diff = 999999.0
        for _ in range(5):
            rand_indices = torch.randperm(candidate_cover_samples.size(0))
            selected_indices = rand_indices[:320]
            cover_samples = candidate_cover_samples[selected_indices]
            assert len(cover_samples) == 320, "cover set size is not 300"

            i = 0
            while i * batch_size < len(cover_samples):
                batch_index = cover_samples[i * batch_size:(i + 1) * batch_size]
                x, y = self.data_reader.get_batch(batch_index)
                out = self.model(x)
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
            cover_gradient = self.get_flatten_parameters() - cache
        
            # 组合cover和poison
            if RESERVED_SAMPLE != 0:
                gradient, indices = select_by_threshold(cover_gradient*cover_factor+gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD) # computed the malicious gradient
            else:
                gradient, indices = select_by_threshold(gradient,GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)

            # limitation
            min_diff = 10.0
            for k in range(len(grad_honest)):
                theta = self.get_cos(gradient, grad_honest[k])
                min_diff = min(theta, min_diff)

            if min_diff >= min_honest_cos_diff:
                if cur_max_agrEvader_grad == None:
                    history_min_diff = min_diff
                    cur_max_agrEvader_grad = gradient
                # elif torch.norm(gradient) > torch.norm(cur_max_agrEvader_grad):
                elif min_diff > history_min_diff:
                    history_min_diff = min_diff
                    print("find a better gradient!")
                    cur_max_agrEvader_grad = gradient

        # 判断使用新生成的还是使用历史最佳的
        if cur_max_agrEvader_grad == None:
            self.aggregator.collect(gradient / torch.norm(gradient), indices)
            # self.aggregator.collect(gradient, indices)
            return gradient
        else :
            self.aggregator.collect(cur_max_agrEvader_grad / torch.norm(cur_max_agrEvader_grad), indices)
            # self.aggregator.collect(cur_max_agrEvader_grad, indices)
            return cur_max_agrEvader_grad

    def evaluate_attack_result(self):
        """
        Evaluate the attack result, return the overall accuracy, member accuracy, and precise
        :return: the number of true member, false member, true non-member, false non-member
        """
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        
        # 保留置信度
        confidences = []  # 成员概率（0~1）
        labels = []
        # class_confidences = F.softmax(out, dim=1).detach()
        class_confidences = F.softmax(out, dim=1).cpu().detach().numpy()
                
        for i in range(len(self.attack_samples)):
            
            # 真实成员标签
            label = 1 if self.attack_samples[i] in self.data_reader.train_set else 0
            labels.append(label)
            # 成员概率计算
            if prediction[i] == batch_y[i]:
                # 预测正确：成员概率 = 正确类别的置信度
                confidence = class_confidences[i][batch_y[i]]
            else:
                # 预测错误：成员概率 = 1 - 正确类别的置信度（或直接设为0）
                confidence = 1 - class_confidences[i][batch_y[i]]
            confidences.append(confidence)

            
            if prediction[i] == batch_y[i]:
                attack_result.append(1)
            else:
                attack_result.append(0)
            if self.attack_samples[i] in self.data_reader.train_set:
                ground_truth.append(1)
            else:
                ground_truth.append(0)

            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            else:
                false_non_member += 1

        return true_member, false_member, true_non_member, false_non_member, np.array(labels), np.array(confidences)


    def evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        :return: The accuracy rate of members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate.cpu().numpy() / len(batch_y)

    def evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        :return: The accuracy rate of non-members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.non_members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate.cpu().numpy() / len(batch_y)


class GreyBoxMalicious(FederatedModel):
    """
    Representing the malicious participant trying to collect data for a white-box membership inference attack
    """

    def __init__(self, reader: DataReader, aggregator: Aggregator):
        """
        Initialize the malicious participant
        :param reader: Reader for the data
        :param aggregator: Global aggregator
        """
        super(GreyBoxMalicious, self).__init__(reader, aggregator, 0)
        self.members = None
        self.non_members = None
        self.attack_samples = self.get_attack_sample()
        self.descending_samples = None
        self.shuffled_labels = {}
        self.shuffle_labels()
        self.global_gradient = torch.zeros(self.get_flatten_parameters().size())
        self.member_prediction = None


    def train(self, batch_size=BATCH_SIZE):
        """
        Start a white-box training
        :param batch_size: The batch size
        :return: The malicious gradient for normal training round
        """
        cache = self.get_flatten_parameters()
        descending_samples = self.data_reader.reserve_set
        i = 0
        while i * batch_size < len(descending_samples):
            batch_index = descending_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        gradient = self.get_flatten_parameters() - cache
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices)
        return gradient

    def greybox_attack(self, ascent_factor=ASCENT_FACTOR, batch_size=BATCH_SIZE,
                                  mislead=True, mislead_factor=1, cover_factor=1):
        """
        Take one step of gradient ascent, the returned gradient is a combination of ascending gradient, descending
        gradient, and misleading gradient
        :param ascent_factor: The factor of the ascending gradient
        :param batch_size: The batch size
        :param mislead: Whether to perform misleading
        :param mislead_factor: The factor of the misleading gradient
        :param cover_factor: The factor of the descending gradient
        :return: malicious gradient generated
        """
        cache = self.get_flatten_parameters()
        self.descending_samples = self.data_reader.reserve_set
        # Perform gradient ascent for the attack samples
        i = 0
        while i * batch_size < len(self.attack_samples):
            batch_index = self.attack_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        asc_gradient = self.get_flatten_parameters() - cache
        self.load_parameters(cache)
        # Perform gradient descent for the rest of samples
        i = 0
        while i * batch_size < len(self.descending_samples):
            batch_index = self.descending_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        desc_gradient = self.get_flatten_parameters() - cache
        if not mislead:
            return desc_gradient - asc_gradient * ascent_factor

        # mislead labels
        self.load_parameters(cache)
        mislead_gradients = []
        for k in range(len(self.shuffled_labels)):
            i = 0
            while i * batch_size < len(self.attack_samples):
                batch_index = self.attack_samples[i * batch_size:(i + 1) * batch_size]
                x, y = self.data_reader.get_batch(batch_index)
                y = self.shuffled_labels[k][batch_index]
                out = self.model(x)
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
            mislead_gradients.append(self.get_flatten_parameters() - cache)
            self.load_parameters(cache)

        # select the best misleading gradient
        selected_k = 0
        largest_gradient_diff = 0
        for k in range(len(mislead_gradients)):
            diff = (mislead_gradients[k] - asc_gradient).norm()
            if diff > largest_gradient_diff:
                largest_gradient_diff = diff
                selected_k = k

        gradient = cover_factor * desc_gradient - asc_gradient * 1 + mislead_factor * mislead_gradients[selected_k]
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, self.DEVICE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices)

        return gradient

    def get_attack_sample(self, attack_samples=NUMBER_OF_ATTACK_SAMPLES, member_rate=BLACK_BOX_MEMBER_RATE):
        """
        Randomly select a sample from the data set
        :param attack_samples: The number of attack samples
        :param member_rate: The rate of member samples
        :return: shuffled data of attacker samples
        """
        member_count = round(attack_samples * member_rate)
        non_member_count = attack_samples - member_count
        self.members = self.data_reader.train_set.flatten()[
            torch.randperm(len(self.data_reader.train_set.flatten()))[:member_count]]
        self.non_members = self.data_reader.test_set.flatten()[
            torch.randperm(len(self.data_reader.test_set.flatten()))[:non_member_count]]
        return torch.cat([self.members, self.non_members])[torch.randperm(attack_samples)]

    def shuffle_labels(self, iteration=GREY_BOX_SHUFFLE_COPIES):
        """
        Shuffle the labels in several random permutation, to be used as misleading labels
        it will repeat the given iteration times denote as k, k different copies will be saved
        :param iteration: The number of copies
        """
        max_label = torch.max(self.data_reader.labels).item()
        for i in range(iteration):
            shuffled = self.data_reader.labels[torch.randperm(len(self.data_reader.labels))]
            for j in torch.nonzero(shuffled == self.data_reader.labels):
                shuffled[j] = (shuffled[j] + torch.randint(max_label, [1]).item()) % max_label
            self.shuffled_labels[i] = shuffled


    def evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        :return: The accuracy rate of members
        """
        batch_x, batch_y = self.data_reader.get_batch(self.members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate / len(batch_y)

    def evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        :return: The accuracy rate of non-members
        """
        batch_x, batch_y = self.data_reader.get_batch(self.non_members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate / len(batch_y)

    def evaluate_attack_result(self):
        """
        Evaluate the attack result, return the overall accuracy, member accuracy, and precise
        :return: the number of true member, false member, true non-member, false non-member
        """
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices

        for i in range(len(self.attack_samples)):
            if prediction[i] == batch_y[i]:
                attack_result.append(1)
            else:
                attack_result.append(0)
            if self.attack_samples[i] in self.data_reader.train_set:
                ground_truth.append(1)
            else:
                ground_truth.append(0)

            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            else:
                false_non_member += 1

        return true_member, false_member, true_non_member, false_non_member

if __name__ == "__main__":
    model = ResNet20(input_shape=(3, 48, 48), pooling_size=8, output_shape=7)

    x = torch.rand((1, 3, 48, 48))
    y = model(x)

    print(y.shape)