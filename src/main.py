from utils import eval_image
import sys
from datamodule import CapsulePoseDataModule
from models import CapsulePose
import torch
import torchvision
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
import capspose_flags
import numpy as np
import os
from absl import app
from absl import flags
FLAGS = flags.FLAGS


def init_all():
    warnings.filterwarnings("ignore")

    # enable cudnn and its inbuilt auto-tuner to find the best algorithm to use for your hardware
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # useful for run-time
    #torch.backends.cudnn.deterministic = True

    pl.seed_everything(FLAGS.seed)
    torch.cuda.empty_cache()


def main(argv):
    init_all()
    print("Capsules architecture: ", FLAGS.arch)

    if FLAGS.mode == "train":
        dm = CapsulePoseDataModule(FLAGS)
        model = CapsulePose(FLAGS)
        if(FLAGS.resume_training):
            trainer = pl.Trainer(gpus=1, distributed_backend=None, resume_from_checkpoint=os.path.join(
                os.getcwd(), FLAGS.load_checkpoint_dir), max_epochs=10000)
        else:
            trainer = pl.Trainer(gpus=1, distributed_backend=None, max_epochs=10000)
        trainer.fit(model, dm)
    elif FLAGS.mode == "test":
        # Create modules
        dm = CapsulePoseDataModule(FLAGS)
        model = CapsulePose(FLAGS)
        model = model.load_from_checkpoint(os.path.join(
                os.getcwd(), FLAGS.load_checkpoint_dir), FLAGS=FLAGS)
        model.configure_optimizers()

        # Manually run prep methods on DataModule
        dm.prepare_data()
        dm.setup()

        # Run test on validation dataset
        trainer = pl.Trainer(gpus=1, distributed_backend=None, resume_from_checkpoint=os.path.join(
                os.getcwd(), FLAGS.load_checkpoint_dir), max_epochs=10000)
        trainer.test(model, test_dataloaders=dm.val_dataloader())
        print(np.array(model.features).shape)
        np.save('output/features', np.array(model.features))
        
    elif FLAGS.mode == "demo":
        model = CapsulePose(FLAGS)
        model = model.load_from_checkpoint(os.path.join(
                os.getcwd(), FLAGS.load_checkpoint_dir), FLAGS=FLAGS)
        model.configure_optimizers()
        model = model.cuda()

        eval_image(model)


if __name__ == '__main__':
    app.run(main)
