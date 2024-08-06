from dataloader import ImageDatasetLoader
from model import DiseaseClassifier
from plot import Plot_Model
import torch
import torch.nn as nn
import random
from datetime import timedelta


from socket import gethostname
import subprocess
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_initialized, barrier

import os
import time
import sys

import logging

logging.basicConfig(level=logging.INFO)
# print("cuDNN Version: ", torch.backends.cudnn.version())
# print("CUDA Available: ", torch.cuda.is_available())
# print("Number of GPUs: ", torch.cuda.device_count())



# DIR_PATH = "C:/Users/alabi/OneDrive - University of North Carolina at Charlotte/Personal Projects/Machine Learning/ASDID/Datasets/Soybean_ML_orig_20"
DIR_PATH = "../Soybean_ML_orig"

def setup (rank: int, world_size: int):
    MASTER_ADDR = os.environ['MASTER_ADDR']
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(random.randint(10000, 20000))
    MASTER_PORT = os.environ['MASTER_PORT']

    print("MASTER_PORT: ", MASTER_PORT)
    print("MASTER_ADDR: ", MASTER_ADDR)

    # Initialize the process group
    try:
        init_process_group(backend="nccl",
                        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                        # init_method="env://",
                        rank=rank,
                        world_size=world_size,
                        timeout=timedelta(minutes=15))
    except Exception as e:
        logging.error(f"Error initializing process group for rank {rank}: {e}")
        exit(1)


def cleanup():
    """Cleanup the process group after training."""
    try:
        destroy_process_group()
        logging.info("Successfully destroyed process group.")
    except Exception as e:
        logging.error(f"Failed to destroy process group: {str(e)}", exc_info=True)

# def start(world_size, rank, local_rank, backbone, setting, num_workers):
#     # Hyperparameters
#     num_epochs = 4
#     batch_size = 32
#     learning_rate = 0.0001
#     momentum = 0.9
#     # manual_seed = 42

#     #rank = int(os.environ["RANK"])
#     #print("Second Rank: ", rank)
#     setup(world_size, rank, local_rank)
#     gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
#     assert gpus_per_node == torch.cuda.device_count()
#     #print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
#         # f" {gpus_per_node} allocated GPUs per node.", flush=True)

#     #if rank == 0: print(f"Group initialized? {is_initialized()}", flush=True)

#     #local_rank = rank - gpus_per_node * (rank // gpus_per_node)
#     #local_rank = rank % torch.cuda.device_count()
#     torch.cuda.set_device(local_rank)
#     device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

#     print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

#     print(f"Backbone: {backbone}; Setting: {setting}")
#     print("Rank:", rank)
#     print("Local Rank:", local_rank)
#     # Print environment variables for debugging
#     #print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"))
#     #print("MASTER_PORT:", os.environ.get("MASTER_PORT"))
#     #print("RANK:", rank)
#     #print("LOCAL_RANK:", local_rank)
#     print("DEVICE:", device)
#     #print("WORLD_SIZE:", world_size)


#     # dataset = ImageDatasetLoader(DIR_PATH, batch_size=batch_size, rank=rank, world_size=world_size, num_workers=num_workers)
#     # print(dataset.class_names)
#     # print(dataset.dataset_sizes)

#     print("End ***************************************")
#     #model = DiseaseClassifier(backbone=backbone, output_shape=len(dataset.class_names), device=device, load_pretrained=ms["pretrained"], freeze=ms["freeze"], pddd=ms["pddd"])
#     #model = model.to(device)
#     #model = DDP(model, device_ids=[local_rank])
#     #model.set_seeds()

#     #criterion = nn.CrossEntropyLoss()
#     #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

#     # model = DDP(model, device_ids=device_ids)
#     #model.train_model(criterion, optimizer, dataset.dataloaders, num_epochs)
#     #model.save_model()
#     #model.test_model(dataset.dataloaders, dataset.class_names)
    
#     #plot = Plot_Model(backbone=backbone, output_shape=len(dataset.class_names), device=device, pddd=ms["pddd"], load_pretrained=ms["pretrained"], freeze=ms["freeze"])
#     #plot.plot()
#     # model_summary = model.summarize_model()
#     # print(model_summary)
#     destroy_process_group()
#start(world_size, rank, local_rank)
def main(rank, world_size, backbone, ms):
    try:
        setup(rank, world_size)
        logging.info(f"Entering main function with rank {rank}")
        # Delay initialization slightly to allow for staggered startup
        # time.sleep(rank * 2)  # Delay based on rank to avoid contention
        # print("INSIDE MAIN NOW")
        # Hyperparameters
        num_epochs = 4
        batch_size = 32
        learning_rate = 0.0001
        momentum = 0.9
        # print("RANK: ", rank)

        # Print the rank for debugging purposes
        # print(f"Hello from rank {rank} of {world_size} on {gethostname()} for {backbone}'s {ms}")

        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # print("LOCAL RANK: ", local_rank)
        # print("DEVICE: ", device)
        # print(f"Backbone: {backbone}; Setting: {ms}")
        logging.info(f"Rank {rank} (and local rank {local_rank}) using device: {device}, Backbone: {backbone}, Settings: {ms}")

        dataset = ImageDatasetLoader(DIR_PATH, batch_size=batch_size, rank=rank, world_size=world_size)
        logging.info("Got dataset")
        sampler = DistributedSampler(dataset.image_datasets,num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataset.update_dataloaders(sampler)
        logging.info("Set sampler")
        model = DiseaseClassifier(backbone=backbone, output_shape=len(dataset.class_names), device=device, load_pretrained=ms["pretrained"], freeze=ms["freeze"], pddd=ms["pddd"], rank=local_rank)
        model = model.to(device)
        logging.info("Initialized model")
        model = DDP(model, device_ids=[local_rank])
        logging.info("Wrapped model with DDP")
        model.set_seeds()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        # # model = DDP(model, device_ids=device_ids)
        logging.info("Start model training")
        model.train_model(criterion, optimizer, dataset.dataloaders, num_epochs)
        model.save_model()
        logging.info("Saved trained model")
        model.test_model(dataset.dataloaders, dataset.class_names)
        
        plot = Plot_Model(backbone=backbone, output_shape=len(dataset.class_names), device=device, pddd=ms["pddd"], load_pretrained=ms["pretrained"], freeze=ms["freeze"])
        plot.plot()
        # model_summary = model.summarize_model()
        # print(model_summary)
        # Example training loop
        # for epoch in range(5):  # replace 5 with actual number of epochs
        #     logging.info(f"Rank {rank}, Epoch {epoch} training on {device}... for {backbone}: {ms}")

        barrier()

        logging.info(f"Finished training on rank {rank}")
    except Exception as e:
        logging.error(f"An error occured on rank {rank}: {str(e)}.")
    finally:
        # destroy_process_group()
        cleanup()
        logging.info(f"Exiting process on rank {rank}.")
        sys.exit(0)  # Ensure the process exits cleanly



def main22(rank, world_size):
    # Device configuration
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = int(os.environ.get("SLURM_PROCID"))
    #print("Global Rank:", rank)

    #rank = int(os.environ["SLURM_PROCID"])
    #local_rank = int(os.environ.get("SLURM_LOCALID"))
    local_rank = rank % torch.cuda.device_count()
    print("LOCAL RANK", local_rank)

    world_size = int(os.environ.get("SLURM_NTASKS"))
    #print("World_size:", world_size)

    #world_size = int(os.environ["WORLD_SIZE"])
    #print("First print of world size:", world_size)

    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    #print("First print of num_workers: ", num_workers)
    
    # Load dataset

    # dataset = ImageDatasetLoader(DIR_PATH, batch_size=batch_size, rank=rank, world_size=world_size, num_workers=num_workers)

    # Model Training Settings
    model_settings = {"pddd+freeze": {
            "pretrained": False,
            "freeze": True,
            "pddd": True},

            "pddd+unfreeze": {
            "pretrained": False,
            "freeze": False,
            "pddd": True},

            "imagenet-pretrained": {
            "pretrained": True,
            "freeze": False,
            "pddd": False}
        }
    for setting in model_settings:
        ms = model_settings[setting]
    # backbones = ["resnet50", "vgg16", "densenet201"]
        # backbones = ["resnet50", "efficientn
        backbones = ["densenet201"]
        for backbone in backbones:
            print("Starting *******************************************************************************************************")
            # mp.spawn(start, args=(world_size, rank, local_rank, backbone, setting, num_workers), nprocs=world_size)



def spawn_process_for_iteration(world_size, setting, backbone, ms):
    """Spawn processes for each specific iteration."""
    logging.info(f"ABOUT TO START SPAWNING FOR {setting} WITH {backbone} *********************")

    # Ensure proper port usage by changing the MASTER_PORT for each iteration
    # os.environ['MASTER_PORT'] = str(29500 + hash(setting + backbone) % 1000)

    # Debugging: Print the current environment variables
    logging.info(f"Spawning with MASTER_ADDR={os.environ['MASTER_ADDR']} and MASTER_PORT={os.environ['MASTER_PORT']}")
    
    # Spawn processes for each model and backbone combination
    mp.spawn(
        main,
        args=(world_size, backbone, ms),  # Pass correct arguments
        nprocs=world_size,
        join=True  # Ensure processes are synchronized and joined
    )
    logging.info(f"Finished training for {setting} with {backbone}")

    

if __name__ == '__main__':
    # Check if the script is being run as the main module
    if "WORLD_SIZE" not in os.environ:
        print("Error: WORLD_SIZE environment variable not set.")
        exit(1)

    print("WORLD_SIZE FOUND in OS_ENVIRON**********************")

    world_size = int(os.environ["WORLD_SIZE"])
    model_settings = {
        "pddd+freeze": {"pretrained": False, "freeze": True, "pddd": True},
        "pddd+unfreeze": {"pretrained": False, "freeze": False, "pddd": True},
        "imagenet-pretrained": {"pretrained": True, "freeze": False, "pddd": False}
    }

    backbones = ["resnet50", "vgg16", "densenet201"]

    spawn_process_for_iteration(world_size, "pddd+freeze", backbones[0], model_settings["pddd+freeze"])
    # spawn_process_for_iteration(world_size, "pddd+freeze", backbones[1], model_settings["pddd+freeze"])
    # spawn_process_for_iteration(world_size, "pddd+freeze", backbones[2], model_settings["pddd+freeze"])

    # spawn_process_for_iteration(world_size, "pddd+unfreeze", backbones[0], model_settings["pddd+unfreeze"])
    # spawn_process_for_iteration(world_size, "pddd+unfreeze", backbones[1], model_settings["pddd+unfreeze"])
    # spawn_process_for_iteration(world_size, "pddd+unfreeze", backbones[2], model_settings["pddd+unfreeze"])

    # spawn_process_for_iteration(world_size, "imagenet-pretrained", backbones[0], model_settings["imagenet-pretrained"])
    # spawn_process_for_iteration(world_size, "imagenet-pretrained", backbones[1], model_settings["imagenet-pretrained"])
    # spawn_process_for_iteration(world_size, "imagenet-pretrained", backbones[2], model_settings["imagenet-pretrained"])



    # for setting, ms in model_settings.items():
    #     for backbone in backbones:
    #         # Call a separate function for spawning processes for each iteration
    #         spawn_process_for_iteration(world_size, setting, backbone, ms)