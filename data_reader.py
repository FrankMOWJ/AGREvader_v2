import numpy as np
import pandas as pd
from constants import *
from torchvision import datasets, transforms

class DataReader:
    """
    The class to read data set from the given file
    """
    def __init__(self, data_set=CIFAR10, data_distribution='iid', number_clients=9, label_column=LABEL_COL, batch_size=BATCH_SIZE, noniid_bias=0.5,
                reserved=0, device='cuda:0'):
        """
        Load the data from the given data path
        :param path: the path of csv file to load data
        :param label_column: the column index of csv file to store the labels
        :param label_size: The number of overall classes in the given data set
        """
        self.DEVICE = device
        # load the csv file
        self.data_set = data_set
        self.num_class = None
        self.NUMBER_OF_PARTICIPANTS = number_clients
        self.noniid_bias = noniid_bias
        if data_set == LOCATION30:
            path = LOCATION30_PATH
            data_frame = pd.read_csv(path, header=None)
            # extract the label
            self.labels = torch.tensor(data_frame[label_column].to_numpy(), dtype=torch.int64).to(self.DEVICE)
            self.labels -= 1
            data_frame.drop(label_column, inplace=True, axis=1)
            # extract the data
            self.data = torch.tensor(data_frame.to_numpy(), dtype=torch.float).to(self.DEVICE)
            self.num_class = 30
        
        elif data_set == CIFAR10:
            # Define the transformation for the CIFAR-10 data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # Load the CIFAR-10 dataset
            cifar10_dataset_train = datasets.CIFAR10(root=CIFAR10_PATH, train=True, download=True, transform=transform)
            cifar10_dataset_test = datasets.CIFAR10(root=CIFAR10_PATH, train=False, download=True, transform=transform)
            
            # Convert the dataset into tensors for data and labels
            overall_data = []
            overall_label = []
            # 把test放在前面
            for i in range(len(cifar10_dataset_test)):
                overall_data.append(cifar10_dataset_test[i][0])
                overall_label.append(cifar10_dataset_test[i][1])
            for i in range(len(cifar10_dataset_train)):
                overall_data.append(cifar10_dataset_train[i][0])
                overall_label.append(cifar10_dataset_train[i][1])
            
            self.data = torch.stack(overall_data)
            self.labels = torch.tensor(overall_label)
            
            self.num_class = 10

        elif data_set == CIFAR100:
            # Define the transformation for the CIFAR-100 data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
            # Load the CIFAR-100 dataset
            cifar100_dataset_train = datasets.CIFAR100(root=CIFAR100_PATH, train=True, download=True, transform=transform)
            cifar100_dataset_test = datasets.CIFAR100(root=CIFAR100_PATH, train=False, download=True, transform=transform)
    
            # Convert the dataset into tensors for data and labels
            overall_data = []
            overall_label = []
            
            # Put the test data first
            for i in range(len(cifar100_dataset_test)):
                overall_data.append(cifar100_dataset_test[i][0])
                overall_label.append(cifar100_dataset_test[i][1])
            
            for i in range(len(cifar100_dataset_train)):
                overall_data.append(cifar100_dataset_train[i][0])
                overall_label.append(cifar100_dataset_train[i][1])
            
            # Stack data into a tensor and convert labels into a tensor
            self.data = torch.stack(overall_data)
            self.labels = torch.tensor(overall_label)
            
            # Set number of classes to 100
            self.num_class = 100

        elif data_set == MNIST:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            # 加载MNIST数据集
            mnist_dataset_train = datasets.MNIST(root=MNIST_PATH, train=True, download=True, transform=transform)
            mnist_dataset_test = datasets.MNIST(root=MNIST_PATH, train=False, download=True, transform=transform)
            
            # 将数据集转换为张量
            overall_data = []
            overall_label = []
            # 把test放在前面
            for i in range(len(mnist_dataset_test)):
                overall_data.append(mnist_dataset_test[i][0])
                overall_label.append(mnist_dataset_test[i][1])
            for i in range(len(mnist_dataset_train)):
                overall_data.append(mnist_dataset_train[i][0])
                overall_label.append(mnist_dataset_train[i][1])
            
            # 将数据和标签转换为张量
            self.data = torch.stack(overall_data)
            self.labels = torch.tensor(overall_label)
            
            # MNIST有10个类别
            self.num_class = 10
        
        elif data_set == FASHION_MNIST:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Fashion-MNIST是单通道，所以归一化只需要一个均值和标准差
            ])
            
            # 加载Fashion-MNIST数据集
            fashion_mnist_dataset_train = datasets.FashionMNIST(root=FASHION_MNIST_PATH, train=True, download=True, transform=transform)
            fashion_mnist_dataset_test = datasets.FashionMNIST(root=FASHION_MNIST_PATH, train=False, download=True, transform=transform)
            
            # 将数据集转换为张量
            overall_data = []
            overall_label = []
            # 把test放在前面
            for i in range(len(fashion_mnist_dataset_test)):
                overall_data.append(fashion_mnist_dataset_test[i][0])
                overall_label.append(fashion_mnist_dataset_test[i][1])
            for i in range(len(fashion_mnist_dataset_train)):
                overall_data.append(fashion_mnist_dataset_train[i][0])
                overall_label.append(fashion_mnist_dataset_train[i][1])
            
            # 将数据和标签转换为张量
            self.data = torch.stack(overall_data)
            self.labels = torch.tensor(overall_label)
            
            # Fashion-MNIST有10个类别
            self.num_class = 10
          
        elif data_set == PURCHASE100:
            self.num_class = 100
            data = np.load(PURCHASE100_PATH)
            self.data = torch.tensor(data['features'], dtype=torch.float)
            self.labels = torch.tensor(data['labels'], dtype=torch.int64)
            self.labels = torch.argmax(self.labels, dim=1)

        elif data_set == TEXAS100:
            self.num_class = 100
            data = np.load(TEXAS100_PATH)
            self.data = torch.tensor(data['features'], dtype=torch.float)
            self.labels = torch.tensor(data['labels'], dtype=torch.int64)
            self.labels = torch.argmax(self.labels, dim=1)
        
        elif data_set == EMOTION:
            # NOTE: not finished yet
            df = pd.read_csv('path_to_your_csv_file.csv')

            # Split the pixels column into individual pixel values
            df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype='int'))

            # Example: Visualizing the first image
            first_image = df['pixels'][0].reshape(48, 48)
            self.num_class = 7
            
        elif data_set == CINIC10:
            cinic_directory = CINIC10_PATH
            cinic_mean = [0.47889522, 0.47227842, 0.43047404]
            cinic_std = [0.24205776, 0.23828046, 0.25874835]
           
            dataset_train = datasets.ImageFolder(cinic_directory + '/train',
                    transform=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
            dataset_test = datasets.ImageFolder(cinic_directory + '/test',
                    transform=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize(mean=cinic_mean,std=cinic_std)]))

            # 将数据集转换为张量
            overall_data = []
            overall_label = []
            # 把test放在前面
            for i in range(len(dataset_test)):
                overall_data.append(dataset_test[i][0])
                overall_label.append(dataset_test[i][1])
            for i in range(len(dataset_train)):
                overall_data.append(dataset_train[i][0])
                overall_label.append(dataset_train[i][1])
            
            # 将数据和标签转换为张量
            self.data = torch.stack(overall_data)
            self.labels = torch.tensor(overall_label)
            
            self.num_class = 10
        
        elif data_set == GTSRB:
            self.num_class = 43
            transform = transforms.Compose([
                transforms.Resize((48, 48)),   # Resize images to 48x48
                transforms.ToTensor(),         # Convert image to PyTorch tensor
            ])

            # Load the dataset from train and test directories
            train_dataset = datasets.ImageFolder(root=GTSRB_PATH + '/Train/Images', transform=transform)
            test_dataset = datasets.ImageFolder(root=GTSRB_PATH + '/Test', transform=transform)

            #TODO: test-set: 12,630, train-set: 39,209
            overall_data = []
            overall_label = []
            # 把test放在前面
            for i in range(len(test_dataset)):
                overall_data.append(test_dataset[i][0])
                overall_label.append(test_dataset[i][1])
            for i in range(len(train_dataset)):
                overall_data.append(train_dataset[i][0])
                overall_label.append(train_dataset[i][1])
            
            self.data = torch.stack(overall_data)
            self.labels = torch.tensor(overall_label)

        elif data_set == SVHN:
            self.num_class = 10

            transform = transforms.Compose([
                transforms.ToTensor(),  # 将图片转换为Tensor
                transforms.Normalize((0.5,), (0.5,))  # 正则化处理
            ])

            # 加载训练集
            train_dataset = torchvision.datasets.SVHN(root=SVHN_PATH, split='train', download=True, transform=transform)
            test_dataset = torchvision.datasets.SVHN(root=SVHN, split='test', download=True, transform=transform)  

            # 将数据集转换为张量
            overall_data = []
            overall_label = []
            # 把test放在前面
            for i in range(len(mnist_dataset_test)):
                overall_data.append(mnist_dataset_test[i][0])
                overall_label.append(mnist_dataset_test[i][1])
            for i in range(len(mnist_dataset_train)):
                overall_data.append(mnist_dataset_train[i][0])
                overall_label.append(mnist_dataset_train[i][1])
            
            # 将数据和标签转换为张量
            self.data = torch.stack(overall_data)
            self.labels = torch.tensor(overall_label)

        elif data_set == SUN397:
            transform = transforms.Compose([
                transforms.Resize((128, 128)),  # 调整图像大小
                transforms.ToTensor(),  # 将图像转换为张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
            ])
            # NOTE: 数据集本身并没有划分出训练集和测试集
            dataset = datasets.SUN397(root=SUN397_PATH, download=True, transform=transform)

            # 将数据集转换为张量
            overall_data = []
            overall_label = []
            for i in range(len(dataset)):
                overall_data.append(dataset[i][0])
                overall_label.append(dataset[i][1])
            
            # 将数据和标签转换为张量
            self.data = torch.stack(overall_data)
            self.labels = torch.tensor(overall_label)
            
            self.num_class = 397
        
        elif data_set == STL10:
            self.num_class = 10
            # transform = transforms.Compose([
            #     transforms.Resize((96, 96)),   # Resize images to 48x48
            #     transforms.ToTensor(),         # Convert image to PyTorch tensor
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
            # ])
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            # Load the dataset from train and test directories
            train_dataset = datasets.STL10(root=STL10_PATH, split='train', download=True, transform=transform)
            test_dataset = datasets.STL10(root=STL10_PATH, split='test', download=True, transform=transform)

            overall_data = []
            overall_label = []
            # 把test放在前面
            for i in range(len(test_dataset)):
                overall_data.append(test_dataset[i][0])
                overall_label.append(test_dataset[i][1])
            for i in range(len(train_dataset)):
                overall_data.append(train_dataset[i][0])
                overall_label.append(train_dataset[i][1])
            
            self.data = torch.stack(overall_data)
            self.labels = torch.tensor(overall_label)

        else:
            raise ValueError(f'no dataset {data_set}')

        print(f'overall data shape: {self.data.shape}\noverall label shape: {self.labels.shape}')
        self.data = self.data.to(self.DEVICE)
        self.labels = self.labels.to(self.DEVICE)
        
        # # 打印前50个label -> shuffle过
        # for i in range(50):
        #     print(self.labels[i])

        assert reserved != 0, 'cover sample should be not 0'

        # initialize the training and testing batches indices
        self.train_set = None
        self.test_set = None
        overall_size = self.labels.size(0)

        # divide data samples into batches, drop the last bit of data samples to make sure each batch is full sized
        overall_size -= reserved
        overall_size -= overall_size % batch_size
        # 生成一个长度为labels.size(0),范围为[0, label.size(0)]的无重复且打乱的序列
        rand_perm = torch.randperm(self.labels.size(0)).to(self.DEVICE)
        self.reserve_set = rand_perm[overall_size:]
        print(f'cover dataset shape: {self.reserve_set.shape}')
        rand_perm = rand_perm[:overall_size].to(self.DEVICE) #! 除去了coverset的下标
        self.batch_indices = rand_perm.reshape((-1, batch_size)).to(self.DEVICE)
        self.train_test_split()

        print("Data set "+data_set+
            " has been loaded, overall {} records, batch size = {}, testing batches: {}, training batches: {}"
            .format(overall_size, batch_size, self.test_set.size(0), self.train_set.size(0)))
            
        if data_distribution == 'non-iid':
            self.train_set = self.make_noniid_dataset(self.num_class, num_users=self.NUMBER_OF_PARTICIPANTS, batch_size=BATCH_SIZE, bias=self.noniid_bias)
            
    def make_noniid_dataset(self, num_class, num_users=1, batch_size=64, bias=0.5):
        bias_weight = bias
        other_group_size = (1-bias_weight) / (num_class-1)
        worker_per_group = num_users / (num_class) # num_worker=nuser, num_ouputs=nclass
        
        each_worker_data = [[] for _ in range(num_users)]
        each_worker_label = [[] for _ in range(num_users)] 
        print(len(each_worker_data))
        
        noniid_train_set = [[] for _ in range(num_users)]

        for batch_indices in self.train_set:
            for sample in batch_indices:
                x, y = self.data[sample], self.labels[sample]
                upper_bound = (y.item()) * (1-bias_weight) / (num_class-1) + bias_weight # default=0.5
                lower_bound = (y.item()) * (1-bias_weight) / (num_class-1)
                rd = np.random.random_sample()
                if rd > upper_bound:
                    worker_group = int(np.floor((rd - upper_bound) / other_group_size)+y.item()+1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.item()

                # assign a data point to a worker
                rd = np.random.random_sample()
                selected_worker = int(worker_group*worker_per_group + int(np.floor(rd*worker_per_group)))
                if (bias_weight == 0): selected_worker = np.random.randint(num_users)
                
                noniid_train_set[selected_worker].append(sample)   
                
        # 将noniid_train_set展开
        noniid_train_set = [sample for sublist in noniid_train_set for sample in sublist]
        # 将noniid_train_set转换为self.train_set的形状
        noniid_train_set = torch.tensor(noniid_train_set).reshape(self.train_set.shape).to(self.DEVICE)
    
        
        # print('######################################################################')
        # print(f'noniid_train_set shape: {noniid_train_set.shape}')
        # print('######################################################################')
        
        return noniid_train_set
            
    def train_test_split(self, ratio=TRAIN_TEST_RATIO, batch_training=BATCH_TRAINING):
        """
        Split the data set into training set and test set according to the given ratio
        :param ratio: tuple (float, float) the ratio of train set and test set
        :param batch_training: True to train by batch, False will not
        :return: None
        """
        if self.data_set == LOCATION30 or self.data_set == PURCHASE100 or self.data_set == TEXAS100 or self.data_set == CINIC10:
            if batch_training:
                train_count = round(self.batch_indices.size(0) * ratio[0] / sum(ratio))
                self.train_set = self.batch_indices[:train_count].to(self.DEVICE)
                self.test_set = self.batch_indices[train_count:].to(self.DEVICE)
            else:
                train_count = round(self.data.size(0) * ratio[0] / sum(ratio))
                rand_perm = torch.randperm(self.data.size(0)).to(self.DEVICE)
                self.train_set = rand_perm[:train_count].to(self.DEVICE)
                self.test_set = rand_perm[train_count:].to(self.DEVICE)
        elif self.data_set == CIFAR10 or self.data_set == MNIST or self.data_set == FASHION_MNIST or self.data_set == CIFAR100:
            test_count = round(self.batch_indices.size(0) / 6.0) # train: 50000, test: 10000
            self.test_set = self.batch_indices[:test_count].to(self.DEVICE)
            self.train_set = self.batch_indices[test_count:].to(self.DEVICE)

        elif self.data_set == GTSRB:
            test_count = round(self.batch_indices.size(0) / 4.1) 
            self.test_set = self.batch_indices[:test_count].to(self.DEVICE)
            self.train_set = self.batch_indices[test_count:].to(self.DEVICE)

        elif self.data_set == SVHN:
            test_count = round(self.batch_indices.size(0) / 3.8) 
            self.test_set = self.batch_indices[:test_count].to(self.DEVICE)
            self.train_set = self.batch_indices[test_count:].to(self.DEVICE)
        
        elif self.data_set == SUN397:
            test_count = round(self.batch_indices.size(0) / 5.0) 
            self.test_set = self.batch_indices[:test_count].to(self.DEVICE)
            self.train_set = self.batch_indices[test_count:].to(self.DEVICE)
        
        # elif self.data_set == STL10:
        #     # train: 5000, test: 8000
        #     test_count = 8 * round(self.batch_indices.size(0) / 13.0) 
        #     self.test_set = self.batch_indices[:test_count].to(self.DEVICE)
        #     self.train_set = self.batch_indices[test_count:].to(self.DEVICE)
            
        elif self.data_set == STL10:
            test_count = round(self.batch_indices.size(0) / 2.5) 
            self.test_set = self.batch_indices[:test_count].to(self.DEVICE)
            self.train_set = self.batch_indices[test_count:].to(self.DEVICE)


    
    def get_train_set(self, participant_index=0):
        """
        Get the indices for each training batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each training batch
        """
        batches_per_participant = self.train_set.size(0) // self.NUMBER_OF_PARTICIPANTS
        lower_bound = participant_index * batches_per_participant
        upper_bound = (participant_index + 1) * batches_per_participant
        return self.train_set[lower_bound: upper_bound]

    def get_test_set(self, participant_index=0):
        """
        Get the indices for each test batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each test batch
        """
        batches_per_participant = self.test_set.size(0) // self.NUMBER_OF_PARTICIPANTS
        lower_bound = participant_index * batches_per_participant
        upper_bound = (participant_index + 1) * batches_per_participant
        return self.test_set[lower_bound: upper_bound]


    def get_batch(self, batch_indices):
        """
        Get the batch of data according to given batch indices
        :param batch_indices: tensor[BATCH_SIZE], the indices of a particular batch
        :return: tuple (tensor, tensor) the tensor representing the data and labels
        """
        return self.data[batch_indices], self.labels[batch_indices]

    def get_black_box_batch(self, member_rate=BLACK_BOX_MEMBER_RATE, attack_batch_size=NUMBER_OF_ATTACK_SAMPLES):
        """
        Generate batches for black box training
        :param member_rate The rate of member data samples
        :param attack_batch_size the number of data samples allocated to the black-box attacker
        """
        member_count = round(attack_batch_size * member_rate)
        non_member_count = attack_batch_size - member_count
        train_flatten = self.train_set.flatten().to(self.DEVICE)
        test_flatten = self.test_set.flatten().to(self.DEVICE)
        member_indices = train_flatten[torch.randperm(len(train_flatten))[:member_count]].to(self.DEVICE)
        non_member_indices = test_flatten[torch.randperm((len(test_flatten)))[:non_member_count]].to(self.DEVICE)
        result = torch.cat([member_indices, non_member_indices]).to(self.DEVICE)
        result = result[torch.randperm(len(result))].to(self.DEVICE)
        # print(f'attack sample shape: {result.shape}\nmember sample shape: {member_indices.shape}')
        return result, member_indices, non_member_indices
    
if __name__ == "__main__":
    data = np.load(TEXAS100_PATH)
    print(f'{data.files}')
    datas = torch.tensor(data['features'], dtype=torch.float)
    labels = torch.tensor(data['labels'], dtype=torch.int64)
    labels = torch.argmax(labels, dim=1)
    
    print(f'{datas.shape}, {labels.shape}')