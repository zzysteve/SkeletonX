import glob
import os
import argparse
import time

def run(cmd):
    print_log(f"Run command: {cmd}")
    if os.system(cmd) != 0:
        print_log(f"[ERROR] when run '{cmd}'")
        exit(0)

class TrainingTimer:
    # setting a timer, including different stage.
    def __init__(self):
        self.time_dict = {}
    
    def start(self, stage):
        self.time_dict[stage] = time.time()

    def stop(self, stage):
        self.time_dict[stage] = time.time() - self.time_dict[stage]
        # print time in both seconds and hh:mm:ss
        hours = int(self.time_dict[stage]//3600)
        minutes = int(self.time_dict[stage]-hours*3600)//60
        seconds = int(self.time_dict[stage])%60
        print_log(f"{stage}: {self.time_dict[stage]:.2f} s, {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def print_summary(self):
        # print time in both seconds and hh:mm:ss
        total_time = sum(self.time_dict.values())
        hours = int(total_time//3600)
        minutes = int(total_time-hours*3600)//60
        seconds = int(total_time)%60
        print_log(f"Total time: {total_time:.2f} s, {hours:02d}:{minutes:02d}:{seconds:02d}")


def print_log(info):
    # add a timestamp to the log
    info = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {info}"
    with open(log_file, 'a') as f:
        print(info)
        print(info, file=f)

if __name__ == '__main__':
    DEBUG = False
    
    timer = TrainingTimer()
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0,1', help='device id, seperated with comma if multiple devices')
    parser.add_argument('--dataset', type=str, default='ntu', help='dataset name')
    args = parser.parse_args()

    device = args.device
    dataset = args.dataset

    log_file = 'logs/' + time.strftime("%Y-%m-%d-%H-%M-%S") + f'_{dataset}.log'
    os.makedirs('./logs', exist_ok=True)
    print_log("This script is used to repeat the experiment in the paper.")

    original_model_dir = f"results/{dataset}/reprod"
    meta_model_dir = f"results/{dataset}/reprod_meta"


    timer.start("Step 1")
    print_log("Step 1: train the original model")
    if not os.path.exists(os.path.join(original_model_dir, 'best_acc.pt')):
        run(f"CUDA_VISIBLE_DEVICES={device} python main_few_shot_xmix.py --config config/{dataset}/one_shot/st_decouple.yaml --work-dir {original_model_dir} --device 0 1")
    else:
        print_log(f"Skip the training process, the model has been trained in {original_model_dir}")
    timer.stop("Step 1")

    timer.start("Step 2")
    print_log("Step 2: meta training process")
    # initialize the meta training process
    run(f"CUDA_VISIBLE_DEVICES={device} python main_few_shot_xmix_meta.py --config config/{dataset}/one_shot/meta_learning.yaml --work-dir {meta_model_dir} --device 0 1 --weights {os.path.join(original_model_dir, 'best_acc.pt')}")
    timer.stop("Step 2")

    timer.print_summary()
    print_log("Finish the experiment.")