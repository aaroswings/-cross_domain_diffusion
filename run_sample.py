import argparse
import json
from train.Trainer import Trainer
from pathlib import Path
from datetime import datetime
import os

parser = argparse.ArgumentParser(
    prog='DiffusionSample'
)
parser.add_argument('profile')
parser.add_argument('--resume_from', type=int, default=0)
parser.add_argument('--device', default='cuda')

parser.add_argument('--num_batches', type=int, default=1)
parser.add_argument('--sample_quantile_dynamic_clip_q', type=float, default=1.0)
parser.add_argument('--sample_intermediates_every_k_steps', type=int, default=200)
parser.add_argument('--replace_eps_alpha', type=float, default=0.0)
parser.add_argument('--do_scheduled_absolute_xclip', type=bool, default=False)


args = parser.parse_args()

if __name__ == '__main__':
    with open(f'profiles/{args.profile}.json') as f:
        config = json.load(f)
    config['diffusion_config']['sample_quantile_dynamic_clip_q'] = args.sample_quantile_dynamic_clip_q
    config['diffusion_config']['sample_intermediates_every_k_steps'] = args.sample_intermediates_every_k_steps
    config['diffusion_config']['replace_eps_alpha'] = args.replace_eps_alpha
    config['diffusion_config']['do_scheduled_absolute_xclip'] = args.do_scheduled_absolute_xclip
    
    trainer_args = {
        'profile': args.profile,
        'checkpoint_to_resume_from': args.resume_from,
        'max_train_steps': 0,
        'device': args.device,
        'checkpoint_every': 0
    }
    trainer = Trainer(
        # merge dicts and unpack as Trainer constructor arguments
        **{**config, **trainer_args}
    )
    trainer.resume_from_checkpoint()    
    timestamp = timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    save_dir = Path(trainer.save_dir) / f'samples_{timestamp}'
    os.makedirs(save_dir)
    with open(save_dir / 'diffusion_config.json', 'w+') as f:
        json.dump(config['diffusion_config'], f)
    trainer.validation_step(args.num_batches, save_dir=save_dir)