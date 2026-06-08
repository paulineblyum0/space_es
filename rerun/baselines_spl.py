import numpy as np
import os



def get_instance_evals(learner, env, num_instances, algo):
    evals = []
    cur_set, _ = env.get_instance_set()
    for i in range(num_instances):
        env.set_instance_set([i])
        obs = env.reset()
        if algo == "trpo":
            val = learner.policy_pi.value([obs])
        else:
            val = learner.value([obs])
        evals.append(val[0])
    env.set_instance_set(cur_set)
    return np.array(evals)


def order_instances_qvals(learner, env, num_instances, algo):
    evals = get_instance_evals(learner, env, num_instances, algo)
    return np.argsort(evals)


def order_instances_improvement(learner, env, num_instances, algo, last_evals):
    evals = get_instance_evals(learner, env, num_instances, algo)
    improvement = evals - last_evals
    return np.argsort(improvement)[::-1], evals


def order_instances_relative_improvement(
    learner, env, num_instances, algo, last_evals
):
    evals = get_instance_evals(learner, env, num_instances, algo)
    absolute_improvement = evals - last_evals
    relative_improvement = absolute_improvement / last_evals
    return np.argsort(relative_improvement)[::-1], evals






