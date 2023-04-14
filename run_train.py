import argparse
import json
from train.Trainer import Trainer

parser = argparse.ArgumentParser(
    prog='DiffusionTrain'
)
parser.add_argument('profile')
parser.add_argument('--max_steps', type=int, default=1000000)
parser.add_argument('--resume_from', type=int, default=0)
parser.add_argument('--checkpoint_every', type=int, default=10000)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

if __name__ == '__main__':
    with open(f'profiles/{args.profile}.json') as f:
        config = json.load(f)
    trainer_args = {
        'profile': args.profile,
        'checkpoint_to_resume_from': args.resume_from,
        'max_train_steps': args.max_steps,
        'device': args.device,
        'checkpoint_every': args.checkpoint_every
    }
    trainer = Trainer(
        # merge dicts and unpack as Trainer constructor arguments
        **{**config, **trainer_args}
    )

    trainer.fit()
