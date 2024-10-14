import os
from datetime import datetime

dataset = ['CIFAR10'] # 'MNIST', 'FASHION_MNIST', 'CIFAR10', 'Purchase100', 'Texas100', 'Location30', 'CIFAR100', 'CINIC10', 'FER2013','STL10'
# distirbution = ['non-iid', 'iid']
# distirbution = ['iid']
distirbution = ['non-iid']

byz_type = ['None', 'Median', 'Trim', 'Angle-Trim'] # 'Angle-Trim'
# byz_type = ['Median']
# byz_type = ['None']
# byz_type = ['Angle-Trim'] # 'Angle-Trim'

# attack_type = [ 'angle', 'norm'] # , 'norm', 'unit'] # 'angle'
# attack_type = ['norm'] # , 'norm', 'unit'] # 'angle'
attack_type = ['angle'] # , 'norm', 'unit'] # 'angle'

# num_clients = [5, 15, 20, 30]
num_clients = [10]
# num_malicious_client = [2,3,4,5]
# num_malicious_client = [2, 3]
# num_malicious_client = [4,5]
num_malicious_client = [1]

bias = [0.1, 0.3, 0.7, 0.9]

# C = [0.8, 1]
# delay = [5, 0]
C = [0.8]
# C = [1]
# delay = [0]
delay = [5]

max_epoch = 800
train_epoch = 0

# gpu = [1]
# gpu = [2]
# gpu = [3]
gpu = [4]

# number_attack_samples = [100, 200, 400, 500]
number_attack_samples = [300]

for each_dataset in dataset:
    for each_data_distribution in distirbution:
        for each_byz_type in byz_type:
            for each_attack_type in attack_type:
                for each_client in num_clients:
                    for each_malicious in num_malicious_client:
                        for each_C in C:
                            for each_delay in delay:
                                for each_attack_samples in number_attack_samples:
                                    for bias_value in bias:
                                        # max_epoch = 150 if each_data_distribution == 'iid' else 200
                                        # train_epoch = 40 if each_data_distribution == 'iid' else 60
                                        # output_dir = f'./log_{each_dataset}/'
                                        # if each_byz_type == 'Angle-Trim' and each_attack_type == 'angle' and (each_malicious == 2 or each_malicious == 3):
                                        #     continue
                                        # if each_byz_type == 'Median' and each_attack_type == 'norm':
                                        #     continue
                                        # if each_C == 0.8 and each_delay == 5:
                                        #     continue
                                        now = datetime.now() #Current date and time
                                        # TIME_STAMP = now.strftime("%Y%m%d") #Time stamp for logging
                                        # TIME_STAMP = 'no-agr'
                                        TIME_STAMP = 'test'
                                        output_dir = rf'./{each_dataset}_{TIME_STAMP}_noniid_bias/'
                                        suffix = "python blackbox_optimized.py" \
                                            + " --dataset=" + str(each_dataset)  \
                                            + " --defense=" + str(each_byz_type) \
                                            + " --attack=" + str(each_attack_type) \
                                            + " --dist=" + str(each_data_distribution) \
                                            + " --Number_client=" + str(each_client) \
                                            + " --Number_malicious=" + str(each_malicious) \
                                                + " --number_attack_samples=" + str(each_attack_samples) \
                                            + " -C=" + str(each_C) \
                                            + " --delay=" + str(each_delay) \
                                            + " --device=" + str(gpu[0]) \
                                            + " --max_epoch=" + str(max_epoch) \
                                            + " --train_epoch=" + str(train_epoch) \
                                            + " --output_dir=" + str(output_dir) \
                                            + " --bias=" + str(bias_value)
                                        os.system(suffix)

# python blackbox_optimized.py --dataset CIFAR10 --defense Trim --attack angle --dist non-iid -C 0.8 --delay 5 --max_epoch 200 --train_epoch 30
# python blackbox_optimized.py --dataset Location30 --defense Krum --attack angle --dist iid -C 0.8 --delay 5 --max_epoch 500
# python blackbox_optimized.py --dataset Purchase100 --defense Krum --attack angle --dist iid -C 0.8 --delay 5 --max_epoch 500 --device 3 --output_dir ./log_Purchase100/
# python blackbox_optimized.py --dataset FASHION_MNIST --defense None --attack norm --dist iid -C 0.8 --delay 5 --max_epoch 500 --device 2 --output_dir ./log_FASHION_MNIST/
# python blackbox_optimized.py --dataset Texas100 --defense None --attack unit --dist iid -C 0.8 --delay 5 --max_epoch 500 --device 0 --output_dir ./log_Texas100/