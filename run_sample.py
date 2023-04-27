import argparse
import json
from train.Trainer import Trainer

parser = argparse.ArgumentParser(
    prog='DiffusionSample'
)
parser.add_argument('profile')
parser.add_argument('--resume_from', type=int, default=0)
parser.add_argument('--device', default='cuda')
# Todo add different sample settings here
args = parser.parse_args()

if __name__ == '__main__':
    with open(f'profiles/{args.profile}.json') as f:
        config = json.load(f)
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