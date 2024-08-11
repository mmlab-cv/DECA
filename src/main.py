from utils import eval_image
import sys
from datamodule import CapsulePoseDataModule
from models import CapsulePose
import torch
import torchvision
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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
    checkpoint_callback = ModelCheckpoint(
        monitor='Training/loss',
        dirpath=FLAGS.checkpoint_dir,
        filename='DECA_2.3.3_TEST-{epoch:02d}-{train_loss:.2f}',
        save_top_k=5,
        save_last=True,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='Validation/loss',
        min_delta=0.05,
        patience=4,
        verbose=True,
        mode='min',
    )
    if FLAGS.mode == "train":
        dm = CapsulePoseDataModule(FLAGS)
        model = CapsulePose(FLAGS)
        if(FLAGS.resume_training):
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                max_epochs=FLAGS.n_epochs,
                callbacks=[early_stop_callback, checkpoint_callback]
            )
            trainer.fit(model, dm, ckpt_path=FLAGS.load_checkpoint_dir)
        else:
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                max_epochs=FLAGS.n_epochs,
                callbacks=[early_stop_callback, checkpoint_callback]
                )
            trainer.fit(model, dm)
        
    elif FLAGS.mode == "test":
        # Create modules
        dm = CapsulePoseDataModule(FLAGS)
        model = CapsulePose(FLAGS)
        model = CapsulePose.load_from_checkpoint(os.path.join(
                os.getcwd(), FLAGS.load_checkpoint_dir), FLAGS=FLAGS)
        model.configure_optimizers()

        # Manually run prep methods on DataModule
        dm.prepare_data()
        dm.setup('test')

        # Run test on validation dataset
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=10000,
        )

        trainer.test(model, ckpt_path=FLAGS.load_checkpoint_dir, dataloaders=dm.val_dataloader())
        print(np.array(model.features).shape)
        #make sure path exists
        np.save(os.path.join('../output', 'features'), np.array(model.features))
        
    elif FLAGS.mode == "demo":
        model = CapsulePose(FLAGS)
        model = CapsulePose.load_from_checkpoint(os.path.join(
                os.getcwd(), FLAGS.load_checkpoint_dir), FLAGS=FLAGS)
        model.configure_optimizers()
        model = model.cuda()

        eval_image(model, FLAG=FLAGS)


if __name__ == '__main__':
    app.run(main)
