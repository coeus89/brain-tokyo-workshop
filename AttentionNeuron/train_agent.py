import argparse
import gin
import os
import util
import numpy as np
import multiprocessing as mp
import cma
import torch
from numpy import genfromtxt
import matplotlib.pyplot as plt
import copy
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', help='Path to config file.')
    parser.add_argument(
        '--log-dir', help='Directory of logs.')
    parser.add_argument(
        '--load-model', help='Path to model file.')
    parser.add_argument(
        '--population-size', help='Population size.', type=int, default=256)
    parser.add_argument(
        '--num-workers', help='Number of workers.', type=int, default=-1)
    parser.add_argument(
        '--num-gpus', help='Number of GPUs for training.', type=int, default=0)  # Changed default to 1. JK
    parser.add_argument(
        '--max-iter', help='Max training iterations.', type=int, default=10000)
    parser.add_argument(
        '--save-interval', help='Model saving period.', type=int, default=100)  # Changed from 100 to 10
    parser.add_argument(
        '--seed', help='Random seed for evaluation.', type=int, default=42)
    parser.add_argument(
        '--reps', help='Number of rollouts for fitness.', type=int, default=16)
    parser.add_argument(
        '--init-sigma', help='Initial std.', type=float, default=0.1)
    parser.add_argument(
        '--train_dir', help='swarm training directory', type=str, default='TempCartPole')
    parser.add_argument(
        '--num_swarms', help='the number of different solver instances', type=int, default=3)
    parser.add_argument(
        '--num_sub_iter', help='number of sub-iterations applied to each swarm', type=int, default=10)
    config, _ = parser.parse_known_args()
    return config


solution = None
task = None


def worker_init(config_file, device_type, num_devices):
    global task, solution
    gin.parse_config_file(config_file)
    task = util.create_task(logger=None)
    worker_id = int(mp.current_process().name.split('-')[-1])
    device = '{}:{}'.format(device_type, (worker_id - 1) % num_devices)
    solution = util.create_solution(device=device)


def get_fitness(params):
    global task, solution
    params, task_seed, num_rollouts = params
    task.seed(task_seed)
    solution.set_params(params)
    scores = []
    for _ in range(num_rollouts):
        scores.append(task.rollout(solution=solution, evaluation=False))
    return np.mean(scores)


def save_params(solver, solution, model_path):
    solution.set_params(solver.result.xfavorite)
    solution.save(model_path)


def main(config):
    logger = util.create_logger(name='train_log', log_dir=config.log_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir, exist_ok=True)
    util.save_config(config.log_dir, config.config)
    logger.info('Logs and models will be save in {}.'.format(config.log_dir))

    rnd = np.random.RandomState(seed=config.seed)
    # solution = util.create_solution(device='cpu:0')
    solutions = [util.create_solution(device='cpu:0') for x in range(config.num_swarms)]
    # solution = util.create_solution(device= torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # JK
    num_params = solutions[0].get_num_params()
    # print(num_params)  # JK
    if config.load_model is not None:
        solutions[0].load(config.load_model)
        print('Loaded model from {}'.format(config.load_model))
        init_params = solutions[0].get_params()
    else:
        init_params = None
    solver = cma.CMAEvolutionStrategy(
        x0=np.zeros(num_params) if init_params is None else init_params,
        sigma0=config.init_sigma,
        inopts={
            'popsize': config.population_size,
            'seed': config.seed if config.seed > 0 else 42,
            'randn': np.random.randn,
        },
    )
    solvers = [copy.deepcopy(solver) for x in range(config.num_swarms)]

    swarm_files = ['model' + str(x) + '.npz' for x in range(config.num_swarms)]
    model_paths = list(map(os.path.join, [config.train_dir for x in range(config.num_swarms)], swarm_files))
    #  save all the blank models
    for i in range(config.num_swarms):
        save_params(solvers[i], solutions[i], model_paths[i])

    best_fitness = -float('Inf')
    best_so_far = [-float('Inf') for x in range(config.num_swarms)]
    ii32 = np.iinfo(np.int32)
    repeats = [config.reps] * config.population_size

    device_type = 'cpu' if args.num_gpus <= 0 else 'cuda'

    #  For data saving as well as displaying a graph.
    fp = os.path.join(config.train_dir, "FitnessLog.csv")
    if os.path.exists(fp):
        os.remove(fp)
    file = open(fp, "w")
    file.write("{},{}\n".format("Iteration", "Fitness"))



    num_devices = mp.cpu_count() if args.num_gpus <= 0 else args.num_gpus
    with mp.get_context('spawn').Pool(
            initializer=worker_init,
            initargs=(args.config, device_type, num_devices),
            processes=config.num_workers,
    ) as pool:
        for n_iter in range(int(config.max_iter / config.num_sub_iter)):  # For each major iteration
            for j in range(config.num_swarms):
                solutions[j].load(model_paths[j])
            for sub_iter in range(config.num_sub_iter):  # For each sub-iteration
                for i in range(config.num_swarms):  # For each swarm
                    solver = solvers[i]
                    params_set = solver.ask()
                    task_seeds = [rnd.randint(0, ii32.max)] * config.population_size
                    fitnesses = []
                    ss = 0
                    while ss < config.population_size:
                        ee = ss + min(config.num_workers, config.population_size - ss)
                        fitnesses.append(
                            pool.map(func=get_fitness,
                                     iterable=zip(params_set[ss:ee],
                                                  task_seeds[ss:ee],
                                                  repeats[ss:ee]))
                        )
                        ss = ee
                    fitnesses = np.concatenate(fitnesses)
                    if isinstance(solver, cma.CMAEvolutionStrategy):
                        # CMA minimizes.
                        solver.tell(params_set, -fitnesses)
                    else:
                        solver.tell(fitnesses)
                    logger.info(
                        'Iter={0}, Swarm={5}, '
                        'max={1:.2f}, avg={2:.2f}, min={3:.2f}, std={4:.2f}'.format(
                            (n_iter * config.num_sub_iter) + sub_iter,  # n_iter * int(config.max_iter / config.num_sub_iter) + sub_iter,
                            np.max(fitnesses), np.mean(fitnesses),
                            np.min(fitnesses), np.std(fitnesses), i))
                    best_iter_fitness = max(fitnesses)
                    if best_iter_fitness > best_so_far[i]:
                        best_so_far[i] = best_iter_fitness
                        # model_path = os.path.join(config.log_dir, 'best.npz')
                        save_params(solvers[i], solutions[i], model_paths[i])

            # if (n_iter + 1) % config.save_interval == 0:
            best_iter_index = best_so_far.index(max(best_so_far))
            if best_so_far[best_iter_index] > best_fitness:
                best_fitness = best_so_far[best_iter_index]
                #  Save the model
                model_path = os.path.join(config.log_dir,
                                          'best_iter_{}.npz'.format((n_iter + 1) * config.num_sub_iter))
                save_params(solver=solvers[best_iter_index], solution=solutions[best_iter_index],
                            model_path=model_path)
                logger.info('Best model updated, score={}'.format(best_fitness))
                #  Save over all of the models with the best
                for j in range(config.num_swarms):
                    save_params(solvers[best_iter_index], solutions[best_iter_index], model_paths[j])
            file.write("{},{}\n".format((n_iter + 1) * config.num_sub_iter, best_fitness))
    pool.join()
    file.close()
    X = []
    Y = []
    X1 = []
    Y1 = []
    X_label = ""
    Y_label = ""
    with open(os.path.join(config.log_dir, "FitnessLog.csv"), 'r') as datafile:
        plotting = csv.reader(datafile, delimiter=',')
        i = False
        for ROWS in plotting:
            if i == False:
                X_label = str(ROWS[0])
                Y_label = str(ROWS[1])
                i = True
            else:
                X.append(int(ROWS[0]))
                Y.append(float(ROWS[1]))

    with open(os.path.join(config.log_dir, "FitnessLog_Original.csv"), 'r') as datafile:
        plotting = csv.reader(datafile, delimiter=',')
        i = False
        for ROWS in plotting:
            if i is not False:
                X1.append(int(ROWS[0]))
                Y1.append(float(ROWS[1]))
            else:
                i = True
    plt.figure(clear=True)
    plt.plot(X, Y, color="blue", label="Swarm Fitness")
    plt.plot(X1, Y1, color="red", label="Original Fitness")
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.title('Fitness Graph')
    plt.legend()
    plt.show()
    plt.close('all')




if __name__ == '__main__':
    args = parse_args()
    if args.num_workers < 0:
        args.num_workers = mp.cpu_count()
    gin.parse_config_file(args.config)
    if args.num_gpus <= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    main(args)
