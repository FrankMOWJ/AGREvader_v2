import os
from datetime import datetime

dataset = ['Texas100'] # 'MNIST', 'FASHION_MNIST', 'CIFAR10', 'Purchase100', 'Texas100', 'Location30', 'CIFAR100', 'CINIC10', 'FER2013'
distirbution = ['non-iid'] #, 'non-iid'] # 'non-iid', 
byz_type = ['Median'] # , 'Trim'
attack_type = ['norm'] # , 'norm', 'unit'] # 'angle'
num_clients = [10]
num_malicious_client = [2]
C = [0.8]
delay = [5]
max_epoch = 500
train_epoch = 0
gpu = [0]
number_attack_samples = [300]

for each_dataset in dataset:
    for each_data_distribution in distirbution:
        for each_byz_type in byz_type:
            for each_attack_type in attack_type:
                for each_client in num_clients:
                    for each_malicious in num_malicious_client:
                        for each_C in C:
                            for each_delay in delay:
                                for each_attack_samples in number_attack_samples
                                # max_epoch = 150 if each_data_distribution == 'iid' else 200
                                # train_epoch = 40 if each_data_distribution == 'iid' else 60
                                # output_dir = f'./log_{each_dataset}/'
                                now = datetime.now() #Current date and time
                                # TIME_STAMP = now.strftime("%Y%m%d") #Time stamp for logging
                                # TIME_STAMP = 'no-agr'
                                TIME_STAMP = 'test'
                                output_dir = rf'./{each_dataset}_{TIME_STAMP}/'
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
                                    + " --output_dir=" + str(output_dir)
                                os.system(suffix)

# python blackbox_optimized.py --dataset CIFAR10 --defense Trim --attack angle --dist non-iid -C 0.8 --delay 5 --max_epoch 200 --train_epoch 30
# python blackbox_optimized.py --dataset Location30 --defense Krum --attack angle --dist iid -C 0.8 --delay 5 --max_epoch 500
# python blackbox_optimized.py --dataset Purchase100 --defense Krum --attack angle --dist iid -C 0.8 --delay 5 --max_epoch 500 --device 3 --output_dir ./log_Purchase100/
# python blackbox_optimized.py --dataset FASHION_MNIST --defense None --attack norm --dist iid -C 0.8 --delay 5 --max_epoch 500 --device 2 --output_dir ./log_FASHION_MNIST/
# python blackbox_optimized.py --dataset Texas100 --defense None --attack unit --dist iid -C 0.8 --delay 5 --max_epoch 500 --device 0 --output_dir ./log_Texas100/