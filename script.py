import os

each_dataset = 'CIFAR10'
each_byz_type = 'Median' # Fang
each_attack_type = 'angle'
each_data_distribution = 'iid'
try_times = 5
each_C = 1.0
each_delay = 0
gpu = [0]

dataset = ['CIFAR10']
distirbution = ['iid']
byz_type = ['Fang', 'Median']
attack_type = ['unit', 'norm', 'angle']
C = [0.5, 0.8]
delay = [5]

for each_dataset in dataset:
    for each_data_distribution in distirbution:
        for each_byz_type in byz_type:
            for each_attack_type in attack_type:
                for each_C in C:
                    for each_delay in delay:
                        suffix = "python blackbox_optimized.py" \
                            + " --dataset=" + str(each_dataset)  \
                            + " --defense=" + str(each_byz_type) \
                            + " --attack=" + str(each_attack_type) \
                            + " --dist=" + str(each_data_distribution) \
                            + " --cover_times=" + str(try_times) \
                            + " -C=" + str(each_C) \
                            + " --delay=" + str(each_delay) \
                            + " --device=" + str(gpu[0]) 
                        os.system(suffix)