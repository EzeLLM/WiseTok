import itertools, random, multiprocessing, time
from functools import partial

import usolib
from usolib.helpers import *


def runtime_sampled(N, uso):
    n_samples = 200
    lst = [usolib.randomfacet.randomfacet_sample(uso, N) for i in range(n_samples)]
    return sum(lst) / float(n_samples), uso


def runtime_analytic(N, uso):
    #return usolib.randomfacet.randomfacet_analytic(uso, N, cache={}), uso
    return usolib.randomfacet.randomfacet_analytic(uso, N) / float(factorial(N) * 2**N), uso


def print_statistic(usos):
    avg = average(runtime for (runtime, uso) in usos)
    print "avg runtime: %s" % avg
    print "max runtime: %s" % usos[-1][0]
    print "min runtime: %s" % usos[0][0]

    print "fst with max runtime:"
    for i in usos[-1][1].get_edges():
        print i

    print


if __name__ == "__main__":
    K = 4
    n_processes = 4
    usos_itr = usolib.uso.all_by_states(K)
    usos = usolib.fst_helpers.uniq(usos_itr)

    for N in range(6, 10):
        print "N=%s" % N
        #print "testing %d usos" % len(usos)
        t = time.time()
        lst = pmap(runtime_analytic, usos, processes=n_processes, args=(N,))
        print "tested %d usos" % len(lst)
        print "finished in %d seconds" % (time.time() - t)

        lst = sorted(lst)
        print_statistic(lst)
        lst = lst[-int(len(lst)*0.1):]
        usos = [uso for (runtime, uso) in lst]

    with open("/home/vitor/tournament_survivors.txt", "w") as fd:
        for runtime, uso in lst:
            print >> fd, runtime
            for i in uso.get_edges():
                print >> fd, i
            print >> fd


