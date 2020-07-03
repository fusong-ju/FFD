import math


def cosine_schedule(init_value, cur_step, nr_step, warmup=0):
    if warmup > 0 and cur_step < warmup:
        return (cur_step + 1) / warmup * init_value
    t, T = cur_step - warmup, nr_step - warmup
    return 0.5 * (1 + math.cos(t * math.pi / T)) * init_value
