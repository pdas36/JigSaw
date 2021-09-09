from __future__ import division
import ast
import sys
import json
# logfile
# filename = sys.argv[1]
def mini_read(logfile_name):
    ''' 
    Function to return only the best mapping execution data (essentially the baseline)
    Input  -> logfile name 
    Output -> baseline execution dictionary
    '''
    with open(logfile_name) as f:
        for line in f:
            if 'best_baseline_counts_dict' in line:
                start_pos = line.find('{')
                end_pos   = line.find(']')
                best_baseline_counts_dict = []
                best_baseline_counts_dict.append(ast.literal_eval(line[start_pos:end_pos]))
    f.close()
    return best_baseline_counts_dict 

def read_data_from_logfile(logfile_name):
    ''' 
    Function to return all the data from the logfile  
    Input  -> logfile name 
    Output -> required data for reconstruction 
    '''
    with open(logfile_name) as f:
        partial_logical_qubits = []      # list of subsets with qubits measured in each subset 
        ideal_histogram_baseline = []    # execution results on the ideal quantum computer
        ideal_histogram_pp = []          # execution results of each of the CPM [stored in same order as the list of subsets]
        baseline_cnot_counts = []        # baseline cnot count post compile
        pp_cnot_counts = []              # cnot counts in the CPM (same order as the list of subsets)
        baseline_counts_dict = []        # baseline counts from execution on nisq device
        pp_counts_dict = []              # CPM counts from execution on nisq device (in the same order in list as the subset of qubits)
        jigsaw_global_mode_dict = []     # jigsaw global mode counts (baseline re adjusted for half the number of trials
        for line in f:
            if 'best_baseline_counts_dict' in line:
                start_pos = line.find('{')
                end_pos   = line.find(']')
                best_baseline_counts_dict = []
                best_baseline_counts_dict.append(ast.literal_eval(line[start_pos:end_pos]))
            else:
                start_pos = line.find('= ') + 2
                end_pos   = line.find('\n')
                if 'partial_logical_qubits' in line:
                    partial_logical_qubits = ast.literal_eval(line[start_pos:end_pos])
                elif 'ideal_histogram_baseline' in line:
                    ideal_histogram_baseline = []
                    ideal_histogram_baseline.append(ast.literal_eval(line[start_pos:end_pos]))
                elif 'ideal_histogram_pp' in line:
                    ideal_histogram_pp = ast.literal_eval(line[start_pos:end_pos])
                elif 'baseline_cnot_counts' in line:
                    baseline_cnot_counts = ast.literal_eval(line[start_pos:end_pos])
                elif 'pp_cnot_counts' in line:
                    pp_cnot_counts = ast.literal_eval(line[start_pos:end_pos])
                elif 'baseline_counts_dict' in line:
                    baseline_counts_dict = ast.literal_eval(line[start_pos:end_pos])
                elif 'trial_adjustments_baseline_counts' in line:
                    jigsaw_global_mode_dict = ast.literal_eval(line[start_pos:end_pos])
                elif 'counts_for_flip_n_measure' in line:
                    counts_for_flip_n_measure = ast.literal_eval(line[start_pos:end_pos])
                elif 'pp_counts_dict' in line:
                    pp_counts_dict = ast.literal_eval(line[start_pos:end_pos])
    f.close()
	
    data = {'baseline': best_baseline_counts_dict, 'ideal': ideal_histogram_baseline, 'ideal_cpm': ideal_histogram_pp, 'baseline_cnot_counts':baseline_cnot_counts, 'cpm_cnot_counts':pp_cnot_counts, 'cpm': partial_logical_qubits, 'edm': baseline_counts_dict, 'cpm_counts': pp_counts_dict, 'jigsaw_global':jigsaw_global_mode_dict}

    return data 

def get_all_experimental_data(filename):
    logfiles = []
    with open(filename) as f:
        ctr = 0 
        for line in f:
            line = line.replace(' ', '') 
            if(line[0] == '['):
                loglist = line[1:line.find(']')]
                loglist = loglist.replace('\'', '')
                loglist = loglist.split(',')
                logfiles.append(loglist)
        f.close()
        
    return logfiles

