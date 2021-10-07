import hashlib
import json
import os

import numpy as np
import pandas as pd


def load_json_template(json_file, vargs):
    # instantiate the template
    with open(json_file, 'r') as f:
        json_text = f.read()
    for i, arg in enumerate(vargs):
        pattern = '"${}"'.format(i + 1)
        json_text = json_text.replace(pattern, arg)
    # load the now ordinary json file
    default_configuration = json.loads(json_text)
    configuration_dict = dict(default_configuration)
    return configuration_dict


def parse_config_dict(d):
    # goes through the dict instantiating any nodes that represent objects
    result_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if 'obj_class' in v and 'module' in v:
                mod = __import__(v["module"], fromlist=[v["obj_class"]])
                obj_class = getattr(mod, v["obj_class"])
                kwargs = v.get("kwargs", {})
                kwargs = parse_config_dict(kwargs)
                obj = obj_class(**{k: v for k, v in kwargs.items() if not k.endswith('.object_info')})
                result_d[k] = obj
                assert ('name' not in kwargs)
                kwargs['name'] = obj_class.__name__
                if hasattr(obj_class, 'object_info'):
                    result_d[k + '.object_info'] = obj_class.object_info(kwargs)
                else:
                    # TODO: this currently leaves out default values (use Signature class to fill these in)
                    result_d[k + '.object_info'] = kwargs
            else:
                result_d[k] = parse_config_dict(v)
        elif isinstance(v, list):
            parsed_list = []
            list_info = []
            for e in v:
                temp = parse_config_dict({'_': e})
                parsed_e = temp['_']
                parsed_list.append(parsed_e)
                e_info = temp.get('_.object_info', e)
                list_info.append(e_info)
            result_d[k] = parsed_list
            result_d[k + '.object_info'] = list_info
        else:
            result_d[k] = v

    return result_d


def serialize_config_dict(d):
    # inverse of parse_config_dict
    # goes through the setup dict serializing any nodes that represent complex objects
    if isinstance(d, dict):
        result_d = {}
        for k, v in d.items():
            if k.endswith('.object_info'):
                pass
            elif k + '.object_info' in d:
                if isinstance(v, list):
                    result_d[k] = [serialize_config_dict(e) for e in d[k + '.object_info']]
                else:
                    result_d[k] = serialize_config_dict(d[k + '.object_info'])
            elif isinstance(v, dict):
                result_d[k] = serialize_config_dict(v)
            elif isinstance(v, list):
                result_d[k] = [serialize_config_dict(e) for e in v]
            else:
                result_d[k] = v
        return result_d
    else:
        return d


class AlreadyStoredException(Exception):

    def __init__(self, id):
        Exception.__init__(self, 'Experiment (id: {}) already performed! (use overwrite=True to overwrite)'.format(id))


def store_result(root, setup, meta=None, sqt=None, overwrite=False):
    setup = serialize_config_dict(setup)
    json_setup = json.dumps(setup, sort_keys=True)
    m = hashlib.md5()
    m.update(json_setup.encode())
    id = m.hexdigest()
    setup_file = os.path.join(root, id + '.setup')
    if os.path.exists(setup_file) and not overwrite:
        raise AlreadyStoredException(id)
    with open(setup_file, 'w') as fh:
        fh.write(json_setup)
    meta_file = os.path.join(root, id + '.meta')
    if os.path.exists(meta_file):
        os.remove(meta_file)
    if meta:
        with open(meta_file, 'w') as fh:
            json.dump(meta, fh)
    sqt_file = os.path.join(root, id + '.sqt')
    if os.path.exists(sqt_file):
        os.remove(sqt_file)
    if sqt:
        sqt.save(sqt_file)
    return id


def load_result(root, id, lazy=False):
    # return setup, sqt
    setup_file = os.path.join(root, id + '.setup')
    with open(setup_file, 'r') as fh:
        setup = json.load(fh)
    setup['.root'] = root
    meta_file = os.path.join(root, id + '.meta')
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as fh:
            setup['.meta'] = json.load(fh)
    if lazy:
        return setup, lambda: SolutionQualityTrace.load(sqt_file=os.path.join(root, id + '.sqt'))
    else:
        return setup, SolutionQualityTrace.load(sqt_file=os.path.join(root, id + '.sqt'))


def load_results(directory, lazy=False, no_duplicates=True):
    loaded_ids = set()
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith('.sqt'):
                id = name[:-4]
                if not no_duplicates or id not in loaded_ids:
                    loaded_ids.add(id)
                    yield load_result(root, name[:-4], lazy=lazy)


class SQTRecorder:

    def __init__(self, time_label='time', verbose=False):
        self.time_label = time_label
        self._prev_budget = None
        self.row_list = []
        self.column_labels = [self.time_label]
        self.verbose = verbose

    def log(self, budget, **metric_values):
        if self._prev_budget is None:
            for key in sorted(metric_values):
                assert (key != self.time_label)
                self.column_labels.append(key)
        elif budget == self._prev_budget:
            self.row_list.pop()
        else:
            assert (self._prev_budget < budget)
            assert (len(self.column_labels) == len(metric_values) + 1)
        self._prev_budget = budget
        row = {clabel: metric_values[clabel] for clabel in self.column_labels[1:]}
        row[self.time_label] = budget
        if self.verbose:
            print(row)
        self.row_list.append(row)

    def produce(self, budget=-float("inf")):

        final_row_list = self.row_list
        if budget > self._prev_budget:
            last_row = dict(self.row_list[-1])
            last_row[self.column_labels[0]] = budget
            final_row_list = self.row_list + [last_row]
        dframe = pd.DataFrame(data=final_row_list, columns=self.column_labels)
        return SolutionQualityTrace(dframe)


class SolutionQualityTrace:

    def __init__(self, dframe):
        self._dframe = dframe

    def save(self, sqt_file):
        sqt_dir = os.path.dirname(sqt_file)
        assert (not os.path.exists(sqt_file))
        if not os.path.exists(sqt_dir):
            os.makedirs(sqt_dir)
        with open(sqt_file, 'w', newline='') as fh:
            self._dframe.to_csv(fh, header=True, index=False)

    @classmethod
    def load(cls, sqt_file):
        with open(sqt_file, 'r') as fh:
            dframe = pd.read_csv(fh)
        return SolutionQualityTrace(dframe)

    def time_label(self):
        return self._dframe.columns[0]

    def change_time_label(self, new_time_label):
        if new_time_label != self.time_label():
            i = self._dframe.columns.get_loc(new_time_label)
            cols = self._dframe.columns.tolist()
            cols[i] = self.time_label()
            cols[0] = new_time_label
            self._dframe = self._dframe[cols]

    def metrics(self):
        return self._dframe.columns[1:]

    def budgets(self):
        return self._dframe[self.time_label()]

    def metric_values(self, metric):
        if isinstance(metric, str):
            # primary metric
            return self._dframe[metric]
        else:
            # secondary metric
            return metric(self._dframe)

    def metric_values_for_budgets(self, metric, budgets):
        trace_budgets = self.budgets()
        trace_values = self.metric_values(metric)
        assert (budgets[0] >= trace_budgets[0])
        values = np.empty(len(budgets), dtype=np.float)
        j = 0
        for k in range(len(budgets)):
            while j + 1 < len(trace_budgets) and trace_budgets[j + 1] < budgets[k]:
                j += 1
            values[k] = trace_values[j]
        return values


class ResultCache:

    DELIM = ">"

    def __init__(self, result_cache_path):
        self.path = result_cache_path
        self.key_list = []
        self.value_list = []
        self.counter = 0
        # load cache
        open(self.path, 'a').close()
        with open(self.path, 'r') as fh:
            line = fh.readline()
            while line:
                # Get next line from file
                index = line.find(self.DELIM)
                if index == -1:
                    break
                key, value = line[:index], float(line[index+1:])
                self.key_list.append(key)
                self.value_list.append(value)
                line = fh.readline()

    def get(self, key):
        if self.counter < len(self.value_list):
            if self.key_list[self.counter] == key:
                value = self.value_list[self.counter]
                self.counter += 1
                return value
            else:
                raise KeyError("Inconsistent cache! (delete the .cache file!)")
        else:
            return None

    def store(self, key, value):
        assert(self.get(key) is None)
        with open(self.path, 'a') as fh:
            fh.write(str(key)+self.DELIM+repr(value))
            fh.write('\n')