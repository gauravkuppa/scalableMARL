import datetime, json, os, argparse, time
import pickle, tabulate
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import os.path as osp
import torch
from torch.utils.tensorboard import SummaryWriter

__author__ = 'Christopher D Hsu'
__copyright__ = ''
__credits__ = ['Christopher D Hsu']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Christopher D Hsu'
__email__ = 'chsu8@seas.upenn.edu'
__status__ = 'Dev'

def load_pytorch_policy(fpath, fname, model, seed):
    fname = osp.join("runs", fpath.split(os.sep)[-1], f"seed_{seed}",fname)
    assert os.path.exists(fname)
    map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(fname, map_location))
    model.to(map_location)
    
    
    # make function for producing an action given a single state
    def get_action(x, deterministic=True):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            action, _, _, _ = model.get_action_and_value(x)
        return action

    return get_action

def eval_set(num_agents, num_targets):
    agents = np.linspace(num_agents/2, num_agents, num=3, dtype=int)
    targets = np.linspace(num_agents/2, num_targets, num=3, dtype=int)
    params_set = [{'nb_agents':1, 'nb_targets':1},
                  {'nb_agents':4, 'nb_targets':4}]
    for a in agents:
        for t in targets:
            params_set.append({'nb_agents':a, 'nb_targets':t})
    return params_set

class Test:
    def __init__(self):
        pass

    def test(self, args, env, act, torch_threads=1):
        num_envs = 1 # hard-coded, only one env for evaluation
        run_name = args.log_dir.split(os.sep)[-1] + "_eval_at_" + datetime.datetime.now().strftime("%m%d%H%M") 
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.set_num_threads(torch_threads)

        seed = args.seed
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        set_eval = eval_set(args.nb_agents, args.nb_targets)
        if args.eval_type == 'random':
            params_set = [{}]
        elif args.eval_type == 'fixed_nb':
            ## Either manually set evaluation set or auto fill
            params_set = SET_EVAL_v0
            # params_set = set_eval
        else:
            raise ValueError("Wrong evaluation type for ttenv.")

        timelimit_env = env
        while( not hasattr(timelimit_env, '_elapsed_steps')):
            timelimit_env = timelimit_env.env

        #if args.ros_log:
        #    from envs.target_tracking.ros_wrapper import RosLog
        #    ros_log = RosLog(num_targets=args.nb_targets, wrapped_num=args.ros + args.render + args.record + 1)
        

        total_nlogdetcov = []
        total_intruders = []
        for params in params_set:
            ep = 0
            ep_nlogdetcov = [] #'Episode nLogDetCov'
            ep_intruders = []
            time_elapsed = ['Elapsed Time (sec)']

            for ep in tqdm(range(args.nb_test_eps)): # test episode
                ep += 1
                s_time = time.time()
                episode_rew, nlogdetcov, ep_len, intruders = 0, 0, 0, 0
                evaluation_total_uncertainity, evaluation_max_uncertainity = 0, 0
                done = np.array([False for _ in range(num_envs)])
                obs = env.reset() # **params)

                while np.sum(done) != num_envs:
                    #
                    if args.render:
                        env.envs[0].render()
                    #if args.ros_log:
                    #    ros_log.log(env)
                    action_dict = [{} for _ in range(num_envs)]
                    for agent_id, o in obs.items():
                        for env_i in range(o.shape[0]):
                            action_dict[env_i][agent_id] = act(o[env_i]).item() # , deterministic=False)
                    
                    obs, rew, done, info = env.step(action_dict)
                    episode_rew += rew
                    nlogdetcov += info['mean_nlogdetcov']
                    ep_len += 1
                    evaluation_total_uncertainity += info['metrics'][0][0]
                    evaluation_max_uncertainity += info['metrics'][0][1]
                    # from IPython import embed; embed()

                time_elapsed.append(time.time() - s_time)
                ep_nlogdetcov.append(nlogdetcov)
                
                #import pdb; pdb.set_trace()
                print(f"Ep.{ep} - Episode reward : {episode_rew}, Episode nLogDetCov : {nlogdetcov}")
                #for env_i in range(args.num_envs):
                writer.add_scalar(f"charts/episodic_return_env_0_eval", episode_rew, ep)
                writer.add_scalar(f"evaluation_total_uncertainity_env_0_eval", evaluation_total_uncertainity, ep)
                writer.add_scalar(f"evaluation_max_uncertainity_env_0_eval", evaluation_max_uncertainity, ep)
                    
            if args.record :
                [moviewriter.finish() for moviewriter in env.envs[0].moviewriters]
            #if args.ros_log :
            #    ros_log.save(args.log_dir)

            # Stats
            meanofeps = np.mean(ep_nlogdetcov)
            total_nlogdetcov.append(meanofeps)
            # Eval plots and saves
            if args.env == 'setTracking-vGreedy':
                eval_dir = os.path.join(os.path.split(args.log_dir)[0], 'greedy_eval_seed%d_'%(seed)+args.map)
            else:
                eval_dir = os.path.join(os.path.split(args.log_dir)[0], 'eval_seed%d_'%(seed)+args.map)
            model_seed = os.path.split(args.log_dir)[-1]           
            # eval_dir = os.path.join(args.log_dir, 'eval_seed%d_'%(seed)+args.map)
            # model_seed = os.path.split(args.log_fname)[0]
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            # matplotlib.use('Agg')
            f0, ax0 = plt.subplots()
            _ = ax0.plot(ep_nlogdetcov, '.')
            _ = ax0.set_title(args.env)
            _ = ax0.set_xlabel('episode number')
            _ = ax0.set_ylabel('mean nlogdetcov')
            _ = ax0.axhline(y=meanofeps, color='r', linestyle='-', label='mean over episodes: %.2f'%(meanofeps))
            _ = ax0.legend()
            _ = ax0.grid()
            _ = f0.savefig(os.path.join(eval_dir, "%da%dt_%d_eval_"%(args.nb_agents, args.nb_targets, args.nb_test_eps)
                                                    +model_seed+".png"))
            plt.close()
            pickle.dump(ep_nlogdetcov, open(os.path.join(eval_dir,"%da%dt_%d_eval_"%(args.nb_agents, args.nb_targets, args.nb_test_eps))
                                                                    +model_seed+".pkl", 'wb'))

        #Plot over all example episode sets
        f1, ax1 = plt.subplots()
        _ = ax1.plot(total_nlogdetcov, '.')
        _ = ax1.set_title(args.env)
        _ = ax1.set_xlabel('example episode set number')
        _ = ax1.set_ylabel('mean nlogdetcov over episodes')
        _ = ax1.grid()
        _ = f1.savefig(os.path.join(eval_dir,'all_%d_eval'%(args.nb_test_eps)+model_seed+'.png'))
        plt.close()        
        pickle.dump(total_nlogdetcov, open(os.path.join(eval_dir,'all_%d_eval'%(args.nb_test_eps))+model_seed+'%da%dt'%(args.nb_agents,args.nb_targets)+'.pkl', 'wb'))
        writer.close()

SET_EVAL_v0 = [
        {'nb_agents': 1, 'nb_targets': 1}]
""",
        {'nb_agents': 2, 'nb_targets': 1},
        {'nb_agents': 3, 'nb_targets': 1},
        {'nb_agents': 4, 'nb_targets': 1},
        {'nb_agents': 1, 'nb_targets': 2},
        {'nb_agents': 2, 'nb_targets': 2},
        {'nb_agents': 3, 'nb_targets': 2},
        {'nb_agents': 4, 'nb_targets': 2},
        {'nb_agents': 1, 'nb_targets': 3},
        {'nb_agents': 2, 'nb_targets': 3},
        {'nb_agents': 3, 'nb_targets': 3},
        {'nb_agents': 4, 'nb_targets': 3},
        {'nb_agents': 1, 'nb_targets': 4},
        {'nb_agents': 2, 'nb_targets': 4},
        {'nb_agents': 3, 'nb_targets': 4},
        {'nb_agents': 4, 'nb_targets': 4},
]"""
