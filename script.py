import os

each_dataset = 'CIFAR10'
each_byz_type = 'Median' # Fang
each_attack_type = 'angle'
each_data_distribution = 'iid'
try_times = 1
each_C = 1.0
each_delay = 0
gpu = [0]


suffix = "python blackbox_optimized.py" \
    + " --data=" + str(each_dataset)  \
    + " --defense=" + str(each_byz_type) \
    + " --attack=" + str(each_attack_type) \
    + " --dist=" + str(each_data_distribution) \
    + " --cover_times=" + str(try_times) \
    + " -C=" + str(each_C) \
    + " --delay=" + str(each_delay) \
    + " --device=" + str(gpu[0]) 
os.system(suffix)