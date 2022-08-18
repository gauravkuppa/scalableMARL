from asyncio.proactor_events import _ProactorDuplexPipeTransport
import pdb, argparse, os, datetime, json, pickle
import torch
import torch.nn as nn

import gym
from gym import wrappers

from algos.maTT.dql import doubleQlearning
import algos.maTT.core as core
from algos.maTT.decentralized_ppo_one_network import decentralized_ppo

import envs

__author__ = 'Gaurav Kuppa'
__copyright__ = ''
__credits__ = ['Gaurav Kuppa']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Christopher D Hsu'
__email__ = 'chsu8@seas.upenn.edu'
__status__ = 'Dev'


os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

BASE_DIR = os.path.dirname('/'.join(str.split(os.path.realpath(__file__),'/')[:-2]))

## cleanRL
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="scalableMARL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    
    parser.add_argument("--total_timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--reward_type", type=str , default='Mean',
                        help="type of reward structure")
    parser.add_argument("--num_steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal_lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent_coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--ppomodel",type=str,default='PPO',
                        help='choose from PPO,PPOAR,PPOAtt,PPOAttAR')
    parser.add_argument("--attention_reward",type=bool,default=False,
                        help='adds attention based target distribution reward')
    parser.add_argument('--scaled', action='store_true')
    parser.set_defaults(scaled=False)
    parser.add_argument('--continue_training', action='store_true')
    parser.set_defaults(continue_training=False)
    
    
    ## maTT
    parser.add_argument('--env', help='environment ID', default='setTracking-v2')
    parser.add_argument('--map', type=str, default="emptyMed")
    parser.add_argument('--nb_agents', type=int, default=4)
    parser.add_argument('--nb_targets', type=int, default=4)
    parser.add_argument('--mode', choices=['train', 'test', 'test-behavior'], default='train')
    parser.add_argument('--record',type=int, default=0)
    parser.add_argument('--render', type=int, default=0)
    parser.add_argument('--nb_test_eps',type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='./results/maTT')
    parser.add_argument('--log_fname', type=str, default='last_model.pt')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--eval_type', choices=['random', 'fixed_4', 
                                                'fixed_2', 'fixed_nb'], default='fixed_nb')
    parser.add_argument('--torch_threads', type=int, default=1)
    parser.add_argument('--amp', type=int, default=0)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def set_model(env,args):
    if args.ppomodel =='PPO':
        model = core.PPO(env, args)
    elif args.ppomodel == 'PPOAR':
        model = core.PPOAR(env, args)
    elif args.ppomodel == 'PPOAtt':
        model = core.PPOAttention(env, args)
    elif args.ppomodel == 'PPOAttAR':
        raise(NotImplemented)
    return model

def train(save_dir, args, notes=None):
    run_name = save_dir.split(os.sep)[-1]
    save_dir_0 = os.path.join(save_dir, f"seed_{args.seed}")
    if not os.path.exists(save_dir_0):
        os.makedirs(save_dir_0)
    else:
        ValueError("The directory already exists...", save_dir_0)
    assert os.path.exists(save_dir_0)

    env = envs.make(args.env,
                    'ma_target_tracking',
                    render=bool(args.render),
                    record=bool(args.record),
                    directory=save_dir_0,
                    map_name=args.map,
                    num_agents=args.nb_agents,
                    num_targets=args.nb_targets,
                    is_training=True,
                    num_envs=args.num_envs,
                    scaled=args.scaled,
                    reward_type = args.reward_type
                    )
    # Create env function
    # env_fn = lambda : env
    model = set_model(env, args)
    if args.continue_training:
        fname = os.path.join("runs", args.log_dir.split(os.sep)[-1], f"seed_{args.seed}", args.log_fname)
        map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.load_state_dict(torch.load(fname, map_location))
    trained_model = decentralized_ppo(env, model, args, run_name, notes)
    # final model saving  
    torch.save(trained_model.state_dict(), os.path.join("runs", run_name, f'seed_{args.seed}', f'last_model.pt'))
    if not args.track:
        print(f"Finished and saved model for training run={run_name}")

def test(args):
    from algos.maTT.evaluation import Test, load_pytorch_policy
    
    env = envs.make(args.env,
                    'ma_target_tracking',
                    render=bool(args.render),
                    record=bool(args.record),
                    directory=args.log_dir,
                    map_name=args.map,
                    num_agents=args.nb_agents,
                    num_targets=args.nb_targets,
                    is_training=False,
                    num_envs=1,
                    scaled=args.scaled,
                    reward_type = args.reward_type
                    )

    # Load saved policy
    model = set_model(env, args)
    policy = load_pytorch_policy(args.log_dir, args.log_fname, model, args.seed)

    # Testing environment
    Eval = Test()
    Eval.test(args, env, policy)

def testbehavior(args):
    from algos.maTT.evaluation_behavior import TestBehavior, load_pytorch_policy
    import algos.maTT.core_behavior as core_behavior
    
    env = envs.make(args.env,
                    'ma_target_tracking',
                    render=bool(args.render),
                    record=bool(args.record),
                    directory=args.log_dir,
                    map_name=args.map,
                    num_agents=args.nb_agents,
                    num_targets=args.nb_targets,
                    is_training=False,
                    )    

    # Load saved policy
    model_kwargs = dict(dim_hidden=args.hiddens)
    model = core_behavior.DeepSetmodel(env.observation_space, env.action_space, **model_kwargs)
    policy = load_pytorch_policy(args.log_dir, args.log_fname, model)

    # Testing environment
    Eval = TestBehavior()
    Eval.test(args, env, policy)


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        date = datetime.datetime.now().strftime("%m%d%H%M")
        run_name = f"{args.env}__{date}"
        save_dir = os.path.join(args.log_dir, run_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            ValueError("The directory already exists...", save_dir)

        notes = input("Any notes for this experiment? : ")
        f = open(os.path.join(save_dir, "notes.txt"), 'w')
        f.write(notes)
        f.close()

        for _ in range(args.repeat):
            print(f"===== TRAIN A TARGET TRACKING RL AGENT : SEED {args.seed} =====")
            model = train(save_dir, args, notes)
            json.dump(vars(args), open(os.path.join(save_dir, f"seed_{args.seed}", 'learning_prop.json'), 'w'))
            args.seed += 1

    elif args.mode =='test':
        test(args)

    elif args.mode =='test-behavior':
        testbehavior(args)
