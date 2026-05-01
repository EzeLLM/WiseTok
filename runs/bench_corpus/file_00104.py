"""Parent Hyperparameter optimization class

Based heavily off of OpenAI spinningup ExperimentGrid class

Implementes abstract base class for hyperparameter search, which is extended
by other classes
"""
import numpy as np
import os.path as osp
from prettytable import PrettyTable


LINE_WIDTH = 80


def call_experiment(agent_cls, exp_name, seed=0, **kwargs):
    """Run an algorithm with hyperparameters (kwargs), plus configuration

    Arguments
    ---------
    agent : Agent
        callable algorithm function
    exp_name : str
        name for experiment
    seed : int
        random number generator seed
    **kwargs : dict
        all kwargs to pass to algo
    """
    # in case seed not in passed kwargs dict
    kwargs['seed'] = seed
    kwargs["exp_name"] = exp_name

    # print experiment details
    table = PrettyTable()
    print("\nRunning experiment: {}".format(exp_name))
    table.field_names = ["Hyperparameter", "Value"]
    for k, v in kwargs.items():
        table.add_row([k, v])
    print("\n", table, "\n")

    agent = agent_cls(**kwargs)
    agent.train()


class Tuner:
    """Abstract base class for specific hyperparam search algorithms

    Subclasses must implement:
    - _run
    """
    line = "\n" + "-"*LINE_WIDTH + "\n"
    thick_line = "\n" + "="*LINE_WIDTH + "\n"

    def __init__(self, name='', seeds=[0]):
        """
        Arguments
        ---------
        name : str
            name for the experiment. This is used when naming files
        seeds : int or list
            the seeds to use for runs.
            If it is a scalar this is taken to be the number of runs and
            so will use all seeds up to scalar
        """
        assert isinstance(name, str), "Name has to be string"
        assert isinstance(seeds, (list, int)), \
            "Seeds must be a int or list of ints"
        self.name = name
        self.keys = []
        self.vals = []
        self.default_vals = []
        self.shs = []

        if isinstance(seeds, int):
            self.seeds = list(range(seeds))
        else:
            self.seeds = seeds

    def run(self, agent_cls, num_cpu=1):
        """Run the tuner.

        Note assumes:
            1. environment is also passed by user as a hyperparam

        Arguments
        ---------
        agent_cls : Class
            the agent class to run
        num_cpu : int
            number of cpus to use
        """
        self.print_info()
        self._run(agent_cls, num_cpu)

    def _run(self, agent_cls, num_cpu=1):
        raise NotImplementedError

    def add(self, key, vals, shorthand=None, default=None):
        """Add a new hyperparam with given values and optional shorthand name

        Arguments
        ---------
        key : str
            name of the hyperparameter (must match arg name in alg function)
        vals : list
            values for hyperparameter
        shorthand : str, optional
            optional shorthand name for hyperparam (if none, one is made
            from first three letters of key)
        default : variable, optional
            optional default value to use for hyperparam. If not
            provided, defaults to first value in vals list.
        """
        if key == "seed":
            print("Warning: Seeds already added to experiment so ignoring "
                  "this hyperparameter addition.")
            return
        self._check_key(key)
        shorthand = self._handle_shorthand(key, shorthand)
        if not isinstance(vals, list):
            vals = [vals]
        self.keys.append(key)
        self.vals.append(vals)
        self.shs.append(shorthand)
        self.default_vals.append(vals[0] if default is None else default)

    def _check_key(self, key):
        """Checks key is valid. """
        assert isinstance(key, str), "Key must be a string."
        assert key[0].isalnum(), "First letter of key mus be alphanumeric."

    def _handle_shorthand(self, key, shorthand):
        """Handles the creation of shorthands """
        assert shorthand is None or isinstance(shorthand, str), \
            "Shorthand must be None or string."
        if shorthand is None:
            shorthand = "".join(ch for ch in key[:3] if ch.isalnum())
        assert shorthand[0].isalnum(), \
            "Shorthand must start with at least one alphanumeric letter."
        return shorthand

    def print_info(self):
        """Prints a message containing details of tuner (i.e. current
        hyperparameters and their values)
        """
        print(self.thick_line)
        print(f"{self.__class__.__name__} Info:")
        table = PrettyTable()
        table.title = f"Tuner - {self.name}"
        headers = ["key", "values", "shorthand", "default"]
        table.field_names = headers

        data = zip(self.keys, self.vals, self.shs, self.default_vals)
        for k, v, s, d in data:
            v_print = 'dist' if callable(v) else v
            d_print = 'dist' if callable(d) else d
            table.add_row([k, v_print, s, d_print])

        num_exps = self.get_num_exps()

        print("\n", table, "\n")
        print(f"Seeds: {self.seeds}")
        print(f"Total number of variants, ignoring seeds: {num_exps}")
        print("Total number of variants, including seeds: "
              f"{num_exps * len(self.seeds)}")
        print(self.thick_line)

    def get_num_exps(self):
        """Returns total number of experiments, not including seeds, that
        will be run
        """
        return int(np.prod([len(v) for v in self.vals]))

    def print_results(self, results):
        """Prints results in a nice table

        Arguments
        ---------
        results : list
            variant experiment result dicts
        """
        table = PrettyTable()
        table.title = "Final results for all experiments"
        any_res = results[0]
        headers = list(any_res.keys())
        table.field_names = headers
        for var_result in results:
            row = []
            for k in headers:
                row.append(var_result[k])
            table.add_row(row)
        print("\n{table}\n")

    def write_results(self, results, data_dir):
        """Writes results to file

        Arguments
        ---------
        results : list
            list of variant experiment result dicts
        data_dir : str
            directory to store data, if None uses current working directory
        """
        output_fname = self.name + "_results.txt"
        if data_dir is not None:
            output_fname = osp.join(data_dir, output_fname)

        headers = list(results[0].keys())
        header_row = "\t".join(headers) + "\n"
        with open(output_fname, "w") as fout:
            fout.write(header_row)
            for var_result in results:
                row = []
                for k in headers:
                    v = var_result[k]
                    vstr = "%.3g" % v if isinstance(v, float) else str(v)
                    row.append(vstr)
                fout.write("\t".join(row) + "\n")

    def name_variant(self, variant):
        """Get the name of variant, where the names is the HPGridTuner
        name followed by shorthand of each hyperparam and value, all
        seperated by underscores

        e.g.
            gridName_h1_v1_h2_v2_h3_v3 ...

        Except:
            1. does not include hyperparams with only a single value
            2. does not include seed
            3. if value is bool only include hyperparam name if val is true
        """
        var_name = self.name
        for k, v, sh in zip(self.keys, self.vals, self.shs):
            if k != 'seed' and (callable(v) or len(v) > 1):
                variant_val = variant[k]
                if not callable(v) and \
                   all([isinstance(val, bool) for val in v]):
                    var_name += ("_" + sh) if variant_val else ''
                elif callable(v):
                    if isinstance(variant_val, float):
                        val_format = "{:.3f}".format(variant_val)
                    else:
                        val_format = str(variant_val)
                    var_name += ("_" + sh + "_" + str(val_format))
                else:
                    var_name += ("_" + sh + "_" + str(variant_val))
        return var_name

    def _run_variant(self, args):
        """Runs a single hyperparameter setting variant with algo for each
        seed.
        """
        exp_num, exp_name, variant, agent_cls = args
        print(f"{self.thick_line}\n{self.name} experiment "
              f"{exp_num} of {self.num_exps}")
        trial_num = 1
        trial_results = []
        for seed in self.seeds:
            print(f"{self.line}\n>>> Running trial {trial_num} of"
                  f" {len(self.seeds)}")
            variant["seed"] = seed
            var_result = call_experiment(agent_cls,
                                         exp_name,
                                         **variant)
            trial_results.append(var_result)
            trial_num += 1
        print(self.line)
        print(f"{exp_name} experiment complete")
        print(self.thick_line)

    def sort_results(self, results, metric):
        """Sorts results by a given metric """
        sorted_results = sorted(results, key=lambda k: k[metric], reverse=True)
        return sorted_results
