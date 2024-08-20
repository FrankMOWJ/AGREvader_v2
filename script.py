import os

each_dataset = 'CIFAR10'
each_byz_type = 'Median' # Fang
each_attack_type = 'angle'
each_data_distribution = 'iid'
try_times = 5
each_C = 1.0
each_delay = 0
gpu = [3]

dataset = ['Texas100'] # 'MNIST', 'FASHION_MNIST', 'CIFAR10', 'Purchase100', 'Texas100'
distirbution = ['iid'] #, 'non-iid'] # 'non-iid', 
byz_type = ['Fang', 'Krum', 'Multi-Krum', 'Deepsight', 'Rflbat']  # 'Flame', 'Foolsgold']
byz_type1 = ['Fang', 'Krum', 'Multi-Krum'] # 'Fang', 'Trim', 'Median'
byz_type2 = ['Deepsight', 'Rflbat', 'Flame', 'Foolsgold']
attack_type = ['angle', 'norm', 'unit']
C = [0.8]
delay = [5]
max_epoch = 500
train_epoch = 200

for each_dataset in dataset:
    for each_data_distribution in distirbution:
        for each_byz_type in byz_type:
            for each_attack_type in attack_type:
                for each_C in C:
                    for each_delay in delay:
                        # max_epoch = 150 if each_data_distribution == 'iid' else 200
                        # train_epoch = 40 if each_data_distribution == 'iid' else 60
                        output_dir = f'./log_{each_dataset}/'
                        suffix = "python blackbox_optimized.py" \
                            + " --dataset=" + str(each_dataset)  \
                            + " --defense=" + str(each_byz_type) \
                            + " --attack=" + str(each_attack_type) \
                            + " --dist=" + str(each_data_distribution) \
                            + " --cover_times=" + str(try_times) \
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