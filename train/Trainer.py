import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import os
from typing import Optional

from train.LinearWarmupLR import LinearWarmupLR
from train.PairedDataset import PairedDataset
from train.util import save_images
from model.network.EMA import EMA
from model.network.UNet import UNet
from model.diffusion.ConcatVDiffusion import ConcatVDiffusion
from model.diffusion.BlendDiffusion import BlendDiffusion

"""
net_config keys: UNet class parameters
diffusion_config keys: Diffusion class parameters
optim_config keys:
"""

class Trainer(Module):
    def __init__(
        self,
        net_config: dict,
        diffusion_config: dict,
        ema_config: dict,
        optim_config: dict,
        data_config: dict,
        profile: str='diffusion',
        which_diffusion: str = 'ConcatVDiffusion',
        checkpoint_to_resume_from: int = 0,
        max_train_steps: int = 1000000,
        checkpoint_every: int = 10,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = torch.device(device)
        self.profile = profile
        self.max_train_steps = max_train_steps
        self.checkpoint_every = checkpoint_every
        self.net_config = net_config
        self.diffusion_config = diffusion_config
        self.which_diffusion = which_diffusion
        self.ema_config = ema_config
        self.data_config = data_config
        self.optim_config = optim_config
        self.global_step = 0
        self.last_saved_checkpoint = 0
        self.checkpoint_to_resume_from = checkpoint_to_resume_from

        self.reset()
        self.setup()
        
    def reset(self):
        self.writer = None
        self.scaler = None
        self.net = None
        self.diffusion = None
        self.ema = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None

    def resume_from_checkpoint(self):
        self.update_checkpoint_paths()
        if self.checkpoint_to_resume_from == self.last_saved_checkpoint:
            return
        if not self.checkpoint_path.is_file():
            raise ValueError(f"No checkpoint at {self.checkpoint_path} was found to resume from.")
        checkpoint = torch.load(self.checkpoint_path)
        self.global_step = checkpoint['global_batch_idx']
        self.net.load_state_dict(checkpoint['net_state'])
        self.optimizer.load_state_dict(checkpoint['opt_state'])
        self.ema.load_state_dict(checkpoint['ema_state'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_state'])
        self.last_saved_checkpoint = self.checkpoint_to_resume_from
        
    def save_checkpoint(self):
        self.checkpoint_to_resume_from += 1
        self.update_checkpoint_paths()
        os.makedirs(self.save_dir)
        torch.save({
            'trainer_state': self.state_dict(),
            'global_batch_idx': self.global_step,
            'net_state': self.net.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'ema_state': self.ema.state_dict(),
            'lr_state': self.lr_scheduler.state_dict()
        }, self.checkpoint_path)
        self.last_saved_checkpoint = self.checkpoint_to_resume_from

    def on_before_epoch_start(self):
        self.update_checkpoint_paths()

    def setup(self):
        self.update_checkpoint_paths()
        if self.writer is None:
            self.writer = SummaryWriter(self.train_root)
        if self.scaler is None:
            self.scaler = torch.cuda.amp.GradScaler()
        if self.net is None or self.ema is None or self.diffusion is None:
            self.configure_model()
        if self.train_loader is None or self.val_loader is None or self.test_loader is None:
            self.configure_data()
        if self.optimizer is None or self.lr_scheduler is None:
            self.configure_optimizer()
        self.resume_from_checkpoint()


    def on_epoch_end(self):
        self.save_checkpoint()
        if self.which_diffusion == 'ConcatVDiffusion':
            self.validation_step(include_unmasked=True)
        else:
            self.validation_step(include_unmasked=False)

    def update_checkpoint_paths(self):
        self.train_root =  Path(f'./out/{self.profile}')
        self.save_dir = self.train_root / f'check_{self.checkpoint_to_resume_from}'
        self.checkpoint_path = self.save_dir / 'trainer_state.ckpt'

    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(
            [*self.net.parameters()], lr=0.0, betas=self.optim_config['adam_betas'],
            eps=1e-7
        )
        self.lr_scheduler = LinearWarmupLR(
            self.optimizer, 
            self.optim_config['end_warmup_lr'], 
            self.optim_config['end_lr'],
            self.optim_config['warmup_steps'],
            self.optim_config['total_steps'] - self.optim_config['warmup_steps']
        )

    def configure_data(self):
        train_dataset = PairedDataset(
            self.data_config['train_roots'],
            self.data_config['size'],
            self.data_config['channels']
        )
        val_dataset = PairedDataset(
            self.data_config['val_roots'],
            self.data_config['size'],
            self.data_config['channels']
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            self.data_config['batch_size'],
            shuffle=True, 
            num_workers=self.data_config['num_workers'])
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            self.data_config['batch_size'],
            shuffle=False, 
            num_workers=self.data_config['num_workers'])
        self.test_loader = None # todo

    def configure_model(self):
        self.net = UNet(**self.net_config).to(self.device)
        if self.which_diffusion == 'ConcatVDiffusion':
            self.diffusion = ConcatVDiffusion(**self.diffusion_config).to(self.device)
        elif self.which_diffusion == 'BlendDiffusion':
            self.diffusion = BlendDiffusion(**self.diffusion_config).to(self.device)
        self.ema = EMA(
            self.net, None, 
            self.ema_config['ema_beta'],
            self.ema_config['ema_update_after_step'], 
            self.ema_config['ema_update_every']
        )

    def batch_to_device(self, batch):
        x0, x0_mask = batch
        x0, x0_mask = x0.to(self.device), x0_mask.to(self.device)
        return x0, x0_mask

    def take_train_steps(self, steps):
        print(f'Training for {steps} steps.')
        self.net.train()
        data_iter = iter(self.train_loader)
        for i in tqdm(range(steps)):
            try:
                x0, x0_mask = self.batch_to_device(next(data_iter))
            except StopIteration:
                data_iter = iter(self.train_loader)
                x0, x0_mask = self.batch_to_device(next(data_iter))

            self.forward_backward(x0)

    def validation_step(
            self, 
            num_batches: int = 1,
            save_dir: Optional[str] = None, 
            include_unmasked: bool = True,
        ):
        # Make a validation sample.
        val_iter = iter(self.val_loader)
        self.net.eval()
        save_dir = Path(save_dir) if save_dir is not None else self.save_dir
        try:
            for batch_i in range(num_batches):
                xs_mask, zs_mask = self.get_validation_sample_batch(val_iter=val_iter, pass_x0_mask=True)
                self.save_samples(xs_mask, self.save_dir / f'{batch_i}_xs_mask')
                self.save_samples(zs_mask, self.save_dir / f'{batch_i}_zs_mask')

                if include_unmasked:
                    xs_none, zs_none = self.get_validation_sample_batch(val_iter=val_iter, pass_x0_mask=False)
                    self.save_samples(xs_none, self.save_dir / f'{batch_i}_xs_none')
                    self.save_samples(zs_none, self.save_dir / f'{batch_i}_zs_none')
        except StopIteration:
            return

    def get_validation_sample_batch(self, val_iter, pass_x0_mask=True):
        x0, x0_mask = self.batch_to_device(next(val_iter))
        x0_mask = None if not pass_x0_mask else x0_mask
        xs, zs = self.diffusion.sample(
                net=self.ema.ema_model, 
                x0=x0,
                x0_mask=x0_mask)
        return xs, zs

    def save_samples(self, samples, path):
        os.makedirs(path)
        # kinda nasty
        channels_A = self.data_config['channels'][0]
        save_images(samples, path, channels_A)

    def forward_backward(self, x0):
        self.lr_scheduler.update()
        self.optimizer.zero_grad()

        loss = self.diffusion.loss(self.net, x0)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.update()

        self.writer.add_scalar('Loss/train', loss, self.global_step)
        self.writer.add_scalar('lr/train', self.lr_scheduler.get_lr(), self.global_step)
        self.global_step += 1

        if torch.isnan(loss):
            raise ValueError('Encountered NaN loss, stopping training.')

    def on_train_start(self):
        self.configure_data()
        self.configure_model()
        self.configure_optimizer()
        self.resume_from_checkpoint()

    def fit(self):
        self.on_train_start()
        for checkpoint_step in range(self.global_step, self.max_train_steps, self.checkpoint_every):
            print(f'====[ {checkpoint_step} / {self.max_train_steps} ]====')
            self.on_before_epoch_start()
            self.take_train_steps(self.checkpoint_every)
            self.on_epoch_end()


