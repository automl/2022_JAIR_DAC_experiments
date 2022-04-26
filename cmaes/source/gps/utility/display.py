import numpy as np
import json
from gps.proto.gps_pb2 import CUR_LOC, ACTION, CUR_PS, CUR_SIGMA, PAST_SIGMA, PAST_OBJ_VAL_DELTAS, PAST_LOC_DELTAS

from datetime import datetime
class Display(object):

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self._log_filename = self._hyperparams['log_filename']
        self._plot_filename = self._hyperparams['plot_filename']
        self._first_update = True

    def _output_column_titles(self, algorithm, policy_titles=False):
        """
        Setup iteration data column titles: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        condition_titles = '%3s | %8s %12s' % ('', '', '')
        itr_data_fields  = '%3s | %8s %12s' % ('itr', 'avg_cost', 'avg_pol_cost')
        for m in range(algorithm.M):
            condition_titles += ' | %8s %9s %-7d' % ('', 'condition', m)
            itr_data_fields  += ' | %8s %8s %8s' % ('  cost  ', '  step  ', 'entropy ')
            condition_titles += ' %8s %8s %8s' % ('', '', '')
            itr_data_fields  += ' %8s %8s %8s %s ' % ('pol_cost', 'kl_div_i', 'kl_div_f', 'samples')
        self.append_output_text(condition_titles)
        self.append_output_text(itr_data_fields)

    def eval(self, sample, cur_cond_idx):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        # cur_fcn = sample.agent.fcns[cur_cond_idx]['fcn_obj']


        final_l = np.zeros(T)

        x = sample.get(CUR_LOC)
        sigma_ = sample.get(CUR_SIGMA)
        sigma = [sigma_[i][0] for i in range(sigma_.shape[0])]
        _, dim = x.shape


        for t in range(T):
            final_l[t] = sample.trajectory[t]

        return x, sigma, final_l

    def get_sample_data(self, sample, cur_cond_idx):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        cur_fcn = sample.agent.fcns[cur_cond_idx]['fcn_obj']

        ps_ = sample.get(CUR_PS)
        ps = [ps_[i][0] for i in range(ps_.shape[0])]
        past_sigma = sample.get(PAST_SIGMA)
        past_obj_val_deltas = sample.get(PAST_OBJ_VAL_DELTAS)
        past_loc_deltas = sample.get(PAST_LOC_DELTAS)

        return ps, past_sigma, past_obj_val_deltas, past_loc_deltas

    def _update_iteration_data(self, algorithm, test_idx, test_fcns, pol_sample_lists, traj_sample_lists):
        """
        Update iteration data information: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        for m,idx in enumerate(test_idx):
            samples = len(pol_sample_lists[m])
            sample = np.random.randint(samples)
            sample_ = 'Sample_' + str(sample)
            test_fcn = test_fcns[m % len(test_fcns)]
            #itr_data = '%s%d' % ('Sample_', i)
            #self.append_output_text(itr_data)
            pol_avg_cost, pol_std, traj_avg_cost, traj_std, pol_avg_sigma, pol_sigma_std, traj_avg_sigma, traj_sigma_std, end_values = self.get_data(pol_sample_lists[m], traj_sample_lists[m], idx)
            self.plot_data(pol_sample_lists[m][0], traj_sample_lists[m][0], test_fcn, pol_avg_cost, traj_avg_cost, pol_avg_sigma, traj_avg_sigma, pol_std, traj_std, pol_sigma_std, traj_sigma_std, end_values)

        return pol_avg_cost

    def get_data(self, pol_samples, traj_samples, cur_cond):
        pol_avg_obj = []
        pol_avg_sigma = []
        traj_avg_obj = []
        traj_avg_sigma = []
        end_values = []
        for m in range(len(pol_samples)):
            _,p_sigma,p_obj_val = self.eval(pol_samples[m], cur_cond)
            _,t_sigma,t_obj_val = self.eval(traj_samples[m], cur_cond)
            pol_avg_obj.append(p_obj_val)
            pol_avg_sigma.append(p_sigma)
            traj_avg_obj.append(t_obj_val)
            traj_avg_sigma.append(t_sigma)
            end_values.append(p_obj_val[-1])
        return np.mean(pol_avg_obj, axis=0), np.std(pol_avg_obj, axis=0), np.mean(traj_avg_obj, axis=0), np.std(traj_avg_obj, axis=0), np.mean(pol_avg_sigma, axis=0), np.std(pol_avg_sigma, axis=0), np.mean(traj_avg_sigma, axis=0), np.std(traj_avg_sigma, axis=0), end_values

    def plot_data(self, pol_sample, traj_sample, cur_cond, pol_costs, traj_costs, pol_sigma, traj_sigma, pol_std, traj_std, pol_sigma_std, traj_sigma_std, end_values):
        log_text = {}
        log_text['Average costs LTO'] = list(pol_costs)
        log_text['Average costs controller'] = list(traj_costs)
        log_text['End values LTO'] = list(end_values)
        log_text['Sigma LTO'] = list(pol_sigma)
        log_text['Sigma controller'] = list(traj_sigma)
        log_text['Std costs LTO'] = list(pol_std)
        log_text['Std costs controller'] = list(traj_std)
        log_text['Std Sigma LTO'] = list(pol_sigma_std)
        log_text['Std Sigma controller'] = list(traj_sigma_std)
        self.append_output_text(log_text, cur_cond)


    def update(self, algorithm, agent, test_fcns, cond_idx_list, pol_sample_lists, traj_sample_lists):

        if self._first_update:
            #self._output_column_titles(algorithm)
            self._first_update = False
        #costs = [np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)]
        pol_costs = self._update_iteration_data(algorithm, test_fcns, cond_idx_list, pol_sample_lists, traj_sample_lists)

        return pol_costs

    def append_output_text(self, text, cur_cond):
        with open('%s_%s.txt' % (self._log_filename, cur_cond), 'a') as f:
            json.dump(text, f)
