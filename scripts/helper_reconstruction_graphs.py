from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import Reconstruction_Functions as RF
import Eval_Metrics as EM
import ast
import get_data_from_logfile as gdata
from scipy.stats.mstats import gmean

from PIL import Image
import matplotlib.image as mpimg
import statistics
import random
import shutil



def perform_reconstruction(benchmarks,logfile_list,num_solutions,outputfilename):
    '''
    Function to perform the Bayesian Reconstruction
    Input  : benchmarks      -> list of benchmarks
             logfile_list    -> list of logfiles where the data is present
             num_solutions   -> number of solutions for the benchmarks
    Output : outputfilename  -> dump the output statistics after reconstruction into a file (this specifies the name) and this data can be used to plot
    '''

    pst_base, pst_edm, pst_jigsaw_no_recomp, pst_recon, pst_recon_hierarchical, ist_base, ist_edm, ist_jigsaw_no_recomp, ist_recon, ist_recon_hierarchical, length_dist, num_qubits = [[] for _ in range(12)]
    rel_pst_edm, rel_pst_jigsaw_no_recomp, rel_pst_recon, rel_pst_recon_hierarchical, rel_ist_edm, rel_ist_jigsaw_no_recomp, rel_ist_recon, rel_ist_recon_hierarchical = [[] for _ in range(8)]
    hdist_base,hdist_edm,hdist_jigsaw,hdist_jigsaw_m,corr_base,corr_edm,corr_jigsaw,corr_jigsaw_m = [[] for _ in range(8)]
    rel_hdist_edm,rel_hdist_jigsaw,rel_hdist_jigsaw_m,rel_corr_edm,rel_corr_jigsaw,rel_corr_jigsaw_m = [[] for _ in range(6)]
    tvd_base,tvd_edm,tvd_jigsaw,tvd_jigsaw_m,rel_tvd_edm,rel_tvd_jigsaw,rel_tvd_jigsaw_m = [[] for _ in range(7)]
    fidelity_base,fidelity_edm,fidelity_jigsaw,fidelity_jigsaw_m,rel_fidelity_edm,rel_fidelity_jigsaw,rel_fidelity_jigsaw_m = [[] for _ in range(7)]
    for entry in range(len(logfile_list)):
        print('Reconstructing for ', benchmarks[entry])
        for fname in logfile_list[entry]:
            if 'len_2' in fname:
                data = gdata.read_data_from_logfile(fname)
                ## verify counts
                # RSL.verify_trial_counts(fname)
                baseline_counts = sum(data['baseline'][0].values())
                num_qubits.append(len(list(data['baseline'][0].keys())[0]))
                length_dist.append(len(data['baseline'][0]))
                pst_base.append(EM.compute_pst(data['ideal'][0],data['baseline'][0],num_solutions[entry]))
                ist_base.append(EM.compute_ist(data['ideal'][0],data['baseline'][0],num_solutions[entry]))
                hdist,corr = EM.compute_hdist(dist_a=data['baseline'][0],dist_b=data['ideal'][0])
                hdist_base.append(hdist)
                tvd = EM.tvd_two_dist(p=data['baseline'][0],q=data['ideal'][0])
                tvd_base.append(tvd)
                fidelity_base.append(1-tvd)
                corr_base.append(corr)
                ## EDM

                ## compute the weights for weighted edm
                #weights_for_edm = []
                #for i in range(len(baseline_counts_dict)):
                #    weight = 0
                #    for j in range(len(baseline_counts_dict)):
                #        weight = weight + EM.Compute_KL_Ideal(dict_in=baseline_counts_dict[i],dict_ideal=baseline_counts_dict[j]) + EM.Compute_KL_Ideal(dict_in=baseline_counts_dict[j],dict_ideal=baseline_counts_dict[i]) 
                #    weights_for_edm.append(weight)
                ## Normalize weights 
                #sum_weights = sum(weights_for_edm)
                #for i in range(len(weights_for_edm)):
                #    weights_for_edm[i] = weights_for_edm[i]/sum_weights

                #edm_dict = {}
                #for i in range(len(baseline_counts_dict)):
                #    edm_dict = helper.weighted_update_dist(edm_dict,baseline_counts_dict[i],weights_for_edm[i])
                edm_dict = {}
                for i in range(4):
                     edm_dict = EM.update_dist(edm_dict,data['edm'][i])
                edm_dict = RF.normalize_dict(edm_dict)
                pst_edm.append(EM.compute_pst(data['ideal'][0],edm_dict,num_solutions[entry]))
                ist_edm.append(EM.compute_ist(data['ideal'][0],edm_dict,num_solutions[entry]))
                hdist,corr = EM.compute_hdist(dist_a=edm_dict,dist_b=data['ideal'][0])
                hdist_edm.append(hdist)
                corr_edm.append(corr)
                tvd = EM.tvd_two_dist(p=edm_dict,q=data['ideal'][0])
                tvd_edm.append(tvd)
                fidelity_edm.append(1-tvd)
                ## Perform Regular Reconstruction 
                marginals =[]
                for i in range(len(data['cpm'])):
                    marginals.append([data['cpm_counts'][i][0],{'Order':data['cpm'][i]}])
                output_reconstructed= RF.recurr_reconstruct(baseline_counts_dict=data['jigsaw_global'][0],
                                         marginal_counts_dict=marginals,
                                         max_recur_count=10)
                pst_recon.append(EM.compute_pst(data['ideal'][0],output_reconstructed,num_solutions[entry]))
                ist_recon.append(EM.compute_ist(data['ideal'][0],output_reconstructed,num_solutions[entry]))

                hdist,corr = EM.compute_hdist(dist_a=output_reconstructed,dist_b=data['ideal'][0])
                hdist_jigsaw.append(hdist)
                corr_jigsaw.append(corr)

                tvd = EM.tvd_two_dist(p=output_reconstructed,q=data['ideal'][0])
                tvd_jigsaw.append(tvd)
                fidelity_jigsaw.append(1-tvd)

                ## Perform Regular Reconstruction with marginals drawn from the global PMF-> corresponds to subsetting alone and no reconstruction
                marginals = []
                for i in range(len(data['cpm_counts'])):
                    ref =  RF.Create_Marginals(orignal_count = data['jigsaw_global'][0], marginal_order = data['cpm'][i])
                    marginals.append([ref,{'Order':data['cpm'][i]}])
                output_reconstructed = RF.recurr_reconstruct(baseline_counts_dict= data['jigsaw_global'][0],
                                         marginal_counts_dict=marginals,
                                         max_recur_count=10)

                pst_jigsaw_no_recomp.append(EM.compute_pst(data['ideal'][0],output_reconstructed,num_solutions[entry]))
                ist_jigsaw_no_recomp.append(EM.compute_ist(data['ideal'][0],output_reconstructed,num_solutions[entry]))
        ## Perform Hierarchical Reconstruction
        for index in range(len(logfile_list[entry])):
            fname = logfile_list[entry][index]
            #print('Reconstruction from ', fname)
            data2 = gdata.read_data_from_logfile(fname)
            #RSL.verify_trial_counts(fname,baseline_counts)
            marginals =[]
            #print(partial_logical_qubits, pp_counts_dict)
            for i in range(len(data2['cpm_counts'])):
                marginals.append([data2['cpm_counts'][i][0],{'Order':data2['cpm'][i]}])
            if(index == 0):
                output_reconstructed_hierarchical = RF.recurr_reconstruct(baseline_counts_dict=data['jigsaw_global'][0],
                                          marginal_counts_dict=marginals,
                                          max_recur_count=10)
            else:
                output_reconstructed_hierarchical = RF.recurr_reconstruct(baseline_counts_dict=output_reconstructed_hierarchical,
                                          marginal_counts_dict=marginals,
                                          max_recur_count=10)
        pst_recon_hierarchical.append(EM.compute_pst(data['ideal'][0],output_reconstructed_hierarchical,num_solutions[entry]))
        ist_recon_hierarchical.append(EM.compute_ist(data['ideal'][0],output_reconstructed_hierarchical,num_solutions[entry])) 
        hdist,corr = EM.compute_hdist(dist_a=output_reconstructed_hierarchical,dist_b=data['ideal'][0])
        hdist_jigsaw_m.append(hdist)
        corr_jigsaw_m.append(corr)
        tvd = EM.tvd_two_dist(p=output_reconstructed_hierarchical,q=data['ideal'][0])
        tvd_jigsaw_m.append(tvd)
        fidelity_jigsaw_m.append(1-tvd)

   	## get the relative statistics for all metrics 
    for i in range(len(pst_base)):
        rel_pst_edm.append(pst_edm[i]/pst_base[i])
        rel_pst_jigsaw_no_recomp.append(pst_jigsaw_no_recomp[i]/pst_base[i])
        rel_pst_recon.append(pst_recon[i]/pst_base[i])
        rel_pst_recon_hierarchical.append(pst_recon_hierarchical[i]/pst_base[i])
        rel_ist_edm.append(ist_edm[i]/ist_base[i])
        rel_ist_jigsaw_no_recomp.append(ist_jigsaw_no_recomp[i]/ist_base[i])
        rel_ist_recon.append(ist_recon[i]/ist_base[i])
        rel_ist_recon_hierarchical.append(ist_recon_hierarchical[i]/ist_base[i])
        rel_hdist_edm.append(hdist_edm[i]/hdist_base[i])
        rel_hdist_jigsaw.append(hdist_jigsaw[i]/hdist_base[i])
        rel_hdist_jigsaw_m.append(hdist_jigsaw_m[i]/hdist_base[i])
        rel_corr_edm.append(corr_edm[i]/corr_base[i])
        rel_corr_jigsaw.append(corr_jigsaw[i]/corr_base[i])
        rel_corr_jigsaw_m.append(corr_jigsaw_m[i]/corr_base[i])
        rel_tvd_edm.append(tvd_edm[i]/tvd_base[i])
        rel_tvd_jigsaw.append(tvd_jigsaw[i]/tvd_base[i])
        rel_tvd_jigsaw_m.append(tvd_jigsaw_m[i]/tvd_base[i])
        rel_fidelity_edm.append(fidelity_edm[i]/fidelity_base[i])
        rel_fidelity_jigsaw.append(fidelity_jigsaw[i]/fidelity_base[i])
        rel_fidelity_jigsaw_m.append(fidelity_jigsaw_m[i]/fidelity_base[i])

    ## write out all the data into a logfile
    f = open(outputfilename,"w+")
    outputdata = {'benchmarks': benchmarks, 'num_qubits': num_qubits, 'length_dist':length_dist, 'pst_baseline': pst_base, 'pst_edm': pst_edm, 'pst_jigsaw_no_recomp':pst_jigsaw_no_recomp, 'pst_jigsaw': pst_recon, 'pst_jigsawm': pst_recon_hierarchical, 'ist_baseline': ist_base, 'ist_edm': ist_edm, 'ist_jigsaw_no_recomp': ist_jigsaw_no_recomp, 'ist_jigsaw': ist_recon, 'ist_jigsawm': ist_recon_hierarchical, 'rel_pst_edm': rel_pst_edm, 'rel_pst_jigsaw_no_recomp': rel_pst_jigsaw_no_recomp, 'rel_pst_jigsaw': rel_pst_recon, 'rel_pst_jigsawm': rel_pst_recon_hierarchical, 'rel_ist_edm': rel_ist_edm, 'rel_ist_jigsaw_no_recomp': rel_ist_jigsaw_no_recomp, 'rel_ist_jigsaw': rel_ist_recon, 'rel_ist_jigsawm': rel_ist_recon_hierarchical, 'hdist_base': hdist_base, 'hdist_edm': hdist_edm, 'hdist_jigsaw': hdist_jigsaw, 'hdist_jigsawm': hdist_jigsaw_m, 'tvd_base': tvd_base, 'tvd_edm': tvd_edm, 'tvd_jigsaw': tvd_jigsaw, 'tvd_jigsawm': tvd_jigsaw_m, 'corr_base': corr_base, 'corr_edm': corr_edm, 'corr_jigsaw': corr_jigsaw, 'corr_jigsawm': corr_jigsaw_m, 'fidelity_base':fidelity_base, 'fidelity_edm': fidelity_edm, 'fidelity_jigsaw': fidelity_jigsaw, 'fidelity_jigsawm': fidelity_jigsaw_m, 'rel_hdist_edm': rel_hdist_edm, 'rel_hdist_jigsaw': rel_hdist_jigsaw, 'rel_hdist_jigsawm': rel_hdist_jigsaw_m, 'rel_tvd_edm': rel_tvd_edm, 'rel_tvd_jigsaw': rel_tvd_jigsaw, 'rel_tvd_jigsawm': rel_tvd_jigsaw_m, 'rel_corr_edm': rel_corr_edm, 'rel_corr_jigsaw': rel_corr_jigsaw, 'rel_corr_jigsawm': rel_corr_jigsaw_m, 'rel_fidelity_edm': rel_fidelity_edm, 'rel_fidelity_jigsaw': rel_fidelity_jigsaw, 'rel_fidelity_jigsawm': rel_fidelity_jigsaw_m} 
    f.write('outputdata = ' + str(outputdata) + '\n')
    f.close()
    print('Done')
    return outputdata 


def get_reconstruction_data_from_outputlogfile(logfile_name):
    '''
    Function to read previously reconstructed data from the logfile to obtain the statistics
    '''
    print('Logfile name ', logfile_name)
    with open(logfile_name) as f:
        for line in f:
            start_pos = line.find('= ') + 2
            end_pos   = line.find('\n')
            outputdata = ast.literal_eval(line[start_pos:end_pos])
    f.close()
    return outputdata
   

def generate_statistics(machines_evaluated,eval_data):
    '''
    Function to generate the mean, max etc. for the paper numbers
    '''
    mean_rel_pst_edm = statistics.mean(eval_data['rel_pst_edm'])
    mean_rel_ist_edm = statistics.mean(eval_data['rel_ist_edm'])
    mean_rel_pst_jigsaw_no_recomp = statistics.mean(eval_data['rel_pst_jigsaw_no_recomp'])
    mean_rel_ist_jigsaw_no_recomp = statistics.mean(eval_data['rel_ist_jigsaw_no_recomp'])
    mean_rel_pst_recon = statistics.mean(eval_data['rel_pst_jigsaw'])
    mean_rel_ist_recon = statistics.mean(eval_data['rel_ist_jigsaw'])
    mean_rel_pst_recon_hierarchical = statistics.mean(eval_data['rel_pst_jigsawm'])
    mean_rel_ist_recon_hierarchical = statistics.mean(eval_data['rel_ist_jigsawm'])

    # compare with edm
    rel_edm_bayesian = np.zeros(len(eval_data['rel_pst_jigsaw']))
    rel_edm_hierarchical = np.zeros(len(eval_data['rel_pst_jigsaw']))
    for i in range(len(eval_data['rel_pst_jigsaw'])):
        rel_edm_bayesian[i] = eval_data['rel_pst_jigsaw'][i]/eval_data['rel_pst_edm'][i]
        rel_edm_hierarchical[i] = eval_data['rel_pst_jigsawm'][i]/eval_data['rel_pst_edm'][i]
    # compare with jigsaw
    rel_jigsaw_hierarchical = np.zeros(len(eval_data['rel_pst_jigsaw']))
    for i in range(len(eval_data['rel_pst_jigsaw'])):
        rel_jigsaw_hierarchical[i] = eval_data['rel_pst_jigsawm'][i]/eval_data['rel_pst_jigsaw'][i]

    print('Mean Rel PST EDM :', mean_rel_pst_edm)
    print('Mean Rel PST JigSaw w/o recompilation :', mean_rel_pst_jigsaw_no_recomp)
    print('Mean Rel PST JigSaw with recompilation :', mean_rel_pst_recon)
    print('Mean Rel PST Recon Hierarchical :', mean_rel_pst_recon_hierarchical)
    print('Max improvement in PST EDM ', max(eval_data['rel_pst_edm']))
    print('Max improvement in PST JigSaw w/o recompilation ', max(eval_data['rel_pst_jigsaw_no_recomp']))
    print('Max improvement in PST JigSaw with recompilation ', max(eval_data['rel_pst_jigsaw']))
    print('Max improvement in PST Recon Hierarchical ', max(eval_data['rel_pst_jigsawm']))
    print('Mean improvement in IST EDM ', mean_rel_ist_edm)
    print('Max improvement in IST EDM ', max(eval_data['rel_ist_edm']))
    print('Mean improvement in IST JigSaw ', mean_rel_ist_recon)
    print('Max improvement in IST Recon EDM ', max(eval_data['rel_ist_jigsaw']))
    print('Mean improvement in IST JigSaw-M ', mean_rel_ist_recon_hierarchical)
    print('Max improvement in IST Recon Hierarchical ', max(eval_data['rel_ist_jigsawm']))
    print(' Comparision ')
    print('Rel Mean wrt EDM (PST) Bayesian ' , mean_rel_pst_recon/mean_rel_pst_edm)
    print('Max wrt EDM (PST) Bayesian ' , max(rel_edm_bayesian))
    print('Rel Mean wrt EDM (PST) Hierarchical ' , mean_rel_pst_recon_hierarchical/mean_rel_pst_edm)
    print('Max wrt EDM (PST) Hierarchical ' , max(rel_edm_hierarchical))
    print('Mean wrt Bayesian (PST) Hierarchical ' , mean_rel_pst_recon_hierarchical/mean_rel_pst_recon)
    print('Max wrt Bayesian (PST) Hierarchical ' , max(rel_jigsaw_hierarchical))
    print('Rel Mean wrt EDM (IST) Bayesian ' , mean_rel_ist_recon/mean_rel_ist_edm)
    print('Rel Mean wrt EDM (IST) Hierarchical ' , mean_rel_ist_recon_hierarchical/mean_rel_ist_edm)
    print('Mean wrt Bayesian (IST) Hierarchical ' , mean_rel_ist_recon_hierarchical/mean_rel_ist_recon)


def get_machine_specific_distribution_metrics_data(machines_evaluated,logdata):
    '''
    Function to compute the statistics for the distance based metrics for different machines
    This is used to shrink the evaluation data presented in the paper
    '''
    ## separate out data for diff machines
    benchmarks_split,hdist_base_split,hdist_edm_split,hdist_jigsaw_split,hdist_jigsaw_m_split,corr_base_split,corr_edm_split,corr_jigsaw_split,corr_jigsaw_m_split = [[[] for _ in range(len(machines_evaluated))] for _ in range(9)]
    tvd_base_split,tvd_edm_split,tvd_jigsaw_split,tvd_jigsaw_m_split,fidelity_base_split,fidelity_edm_split,fidelity_jigsaw_split,fidelity_jigsaw_m_split = [[[] for _ in range(len(machines_evaluated))] for _ in range(8)]
    rel_hdist_edm_split,rel_hdist_jigsaw_split,rel_hdist_jigsaw_m_split,rel_corr_edm_split,rel_corr_jigsaw_split,rel_corr_jigsaw_m_split = [[[] for _ in range(len(machines_evaluated))] for _ in range(6)]
    rel_tvd_edm_split,rel_tvd_jigsaw_split,rel_tvd_jigsaw_m_split,rel_fidelity_edm_split,rel_fidelity_jigsaw_split,rel_fidelity_jigsaw_m_split = [[[] for _ in range(len(machines_evaluated))] for _ in range(6)]

    for entry in range(len(logdata['benchmarks'])):
        for index in range(len(machines_evaluated)):
            if(machines_evaluated[index] in logdata['benchmarks'][entry]):
               entry_id = index
               break
        workload = logdata['benchmarks'][entry][0:logdata['benchmarks'][entry].find('\\')]
        if 'p' in workload:
            workload = workload.replace('p', ' p')
        benchmarks_split[entry_id].append(workload)
        hdist_base_split[entry_id].append(logdata['hdist_base'][entry])
        hdist_edm_split[entry_id].append(logdata['hdist_edm'][entry])
        hdist_jigsaw_split[entry_id].append(logdata['hdist_jigsaw'][entry])
        hdist_jigsaw_m_split[entry_id].append(logdata['hdist_jigsawm'][entry])
        tvd_base_split[entry_id].append(logdata['tvd_base'][entry])
        tvd_edm_split[entry_id].append(logdata['tvd_edm'][entry])
        tvd_jigsaw_split[entry_id].append(logdata['tvd_jigsaw'][entry])
        tvd_jigsaw_m_split[entry_id].append(logdata['tvd_jigsawm'][entry])
        corr_base_split[entry_id].append(logdata['corr_base'][entry])
        corr_edm_split[entry_id].append(logdata['corr_edm'][entry])
        corr_jigsaw_split[entry_id].append(logdata['corr_jigsaw'][entry])
        corr_jigsaw_m_split[entry_id].append(logdata['corr_jigsawm'][entry])
        fidelity_base_split[entry_id].append(logdata['fidelity_base'][entry])
        fidelity_edm_split[entry_id].append(logdata['fidelity_edm'][entry])
        fidelity_jigsaw_split[entry_id].append(logdata['fidelity_jigsaw'][entry])
        fidelity_jigsaw_m_split[entry_id].append(logdata['fidelity_jigsawm'][entry])
        rel_hdist_edm_split[entry_id].append(logdata['rel_hdist_edm'][entry])
        rel_hdist_jigsaw_split[entry_id].append(logdata['rel_hdist_jigsaw'][entry])
        rel_hdist_jigsaw_m_split[entry_id].append(logdata['rel_hdist_jigsawm'][entry])
        rel_corr_edm_split[entry_id].append(logdata['rel_corr_edm'][entry])
        rel_corr_jigsaw_split[entry_id].append(logdata['rel_corr_jigsaw'][entry])
        rel_corr_jigsaw_m_split[entry_id].append(logdata['rel_corr_jigsawm'][entry])
        rel_tvd_edm_split[entry_id].append(logdata['rel_tvd_edm'][entry])
        rel_tvd_jigsaw_split[entry_id].append(logdata['rel_tvd_jigsaw'][entry])
        rel_tvd_jigsaw_m_split[entry_id].append(logdata['rel_tvd_jigsawm'][entry])
        rel_fidelity_edm_split[entry_id].append(logdata['rel_fidelity_edm'][entry])
        rel_fidelity_jigsaw_split[entry_id].append(logdata['rel_fidelity_jigsaw'][entry])
        rel_fidelity_jigsaw_m_split[entry_id].append(logdata['rel_fidelity_jigsawm'][entry])

    # add additional data point for mean
    for entry in range(len(benchmarks_split)):
        benchmarks_split[entry].append('GMean')
        hdist_base_split[entry].append(gmean(hdist_base_split[entry]))
        hdist_edm_split[entry].append(gmean(hdist_edm_split[entry]))
        hdist_jigsaw_split[entry].append(gmean(hdist_jigsaw_split[entry]))
        hdist_jigsaw_m_split[entry].append(gmean(hdist_jigsaw_m_split[entry]))
        corr_base_split[entry].append(gmean(corr_base_split[entry]))
        corr_edm_split[entry].append(gmean(corr_edm_split[entry]))
        corr_jigsaw_split[entry].append(gmean(corr_jigsaw_split[entry]))
        corr_jigsaw_m_split[entry].append(gmean(corr_jigsaw_m_split[entry]))
        tvd_base_split[entry].append(gmean(tvd_base_split[entry]))
        tvd_edm_split[entry].append(gmean(tvd_edm_split[entry]))
        tvd_jigsaw_split[entry].append(gmean(tvd_jigsaw_split[entry]))
        tvd_jigsaw_m_split[entry].append(gmean(tvd_jigsaw_m_split[entry]))
        fidelity_base_split[entry].append(gmean(fidelity_base_split[entry]))
        fidelity_edm_split[entry].append(gmean(fidelity_edm_split[entry]))
        fidelity_jigsaw_split[entry].append(gmean(fidelity_jigsaw_split[entry]))
        fidelity_jigsaw_m_split[entry].append(gmean(fidelity_jigsaw_m_split[entry]))
        rel_hdist_edm_split[entry].append(gmean(rel_hdist_edm_split[entry]))
        rel_hdist_jigsaw_split[entry].append(gmean(rel_hdist_jigsaw_split[entry]))
        rel_hdist_jigsaw_m_split[entry].append(gmean(rel_hdist_jigsaw_m_split[entry]))
        rel_corr_edm_split[entry].append(gmean(rel_corr_edm_split[entry]))
        rel_corr_jigsaw_split[entry].append(gmean(rel_corr_jigsaw_split[entry]))
        rel_corr_jigsaw_m_split[entry].append(gmean(rel_corr_jigsaw_m_split[entry]))
        rel_tvd_edm_split[entry].append(gmean(rel_tvd_edm_split[entry]))
        rel_tvd_jigsaw_split[entry].append(gmean(rel_tvd_jigsaw_split[entry]))
        rel_tvd_jigsaw_m_split[entry].append(gmean(rel_tvd_jigsaw_m_split[entry]))
        rel_fidelity_edm_split[entry].append(gmean(rel_fidelity_edm_split[entry]))
        rel_fidelity_jigsaw_split[entry].append(gmean(rel_fidelity_jigsaw_split[entry]))
        rel_fidelity_jigsaw_m_split[entry].append(gmean(rel_fidelity_jigsaw_m_split[entry]))
   
    split_logfile_data = {}
    split_logfile_data['benchmarks'] = benchmarks_split
    split_logfile_data['hdist'] = {'base': hdist_base_split, 'edm': hdist_edm_split, 'jigsaw': hdist_jigsaw_split, 'jigsawm': hdist_jigsaw_m_split}
    split_logfile_data['rel_hdist'] = {'edm': rel_hdist_edm_split, 'jigsaw': rel_hdist_jigsaw_split, 'jigsawm': rel_hdist_jigsaw_m_split}
    split_logfile_data['corr'] = {'base': corr_base_split, 'edm': corr_edm_split, 'jigsaw': corr_jigsaw_split, 'jigsawm': corr_jigsaw_m_split}
    split_logfile_data['rel_corr'] = {'edm': rel_corr_edm_split, 'jigsaw': rel_corr_jigsaw_split, 'jigsawm': rel_corr_jigsaw_m_split}
    split_logfile_data['tvd'] = {'base': tvd_base_split, 'edm': tvd_edm_split, 'jigsaw': tvd_jigsaw_split, 'jigsawm': tvd_jigsaw_m_split}
    split_logfile_data['rel_tvd'] = {'edm': rel_tvd_edm_split, 'jigsaw': rel_tvd_jigsaw_split, 'jigsawm': rel_tvd_jigsaw_m_split}
    split_logfile_data['fidelity'] = {'base': fidelity_base_split, 'edm': fidelity_edm_split, 'jigsaw': fidelity_jigsaw_split, 'jigsawm': fidelity_jigsaw_m_split}
    split_logfile_data['rel_fidelity'] = {'edm': rel_fidelity_edm_split, 'jigsaw': rel_fidelity_jigsaw_split, 'jigsawm': rel_fidelity_jigsaw_m_split}

    return split_logfile_data

def get_machine_specific_data(machines_evaluated,logdata):
    '''
    Function to compute the statistics for the non distance based metrics for different machines
    This is used to shrink the evaluation data presented in the paper
    '''
    ## separate out data for diff machines
    benchmarks_split, pst_edm_split, pst_jigsaw_no_recomp_split, pst_recon_split, pst_hierarchical_split, ist_edm_split, ist_jigsaw_no_recomp_split, ist_recon_split, ist_hierarchical_split = [[[] for _ in range(len(machines_evaluated))] for _ in range(9)]
    rel_pst_edm_split, rel_pst_jigsaw_no_recomp_split, rel_pst_recon_split, rel_pst_hierarchical_split, rel_ist_edm_split, rel_ist_jigsaw_no_recomp_split, rel_ist_recon_split, rel_ist_hierarchical_split = [[[] for _ in range(len(machines_evaluated))] for _ in range(8)]
    for entry in range(len(logdata['benchmarks'])):
        for index in range(len(machines_evaluated)):
            if(machines_evaluated[index] in logdata['benchmarks'][entry]):
               entry_id = index
               break
        workload = logdata['benchmarks'][entry][0:logdata['benchmarks'][entry].find('\\')]
        if 'p' in workload:
            workload = workload.replace('p', ' p')
        benchmarks_split[entry_id].append(workload+'\n'+str(round(logdata['pst_baseline'][entry],2)))
        pst_edm_split[entry_id].append(logdata['pst_edm'][entry])
        rel_pst_edm_split[entry_id].append(logdata['rel_pst_edm'][entry])
        ist_edm_split[entry_id].append(logdata['ist_edm'][entry])
        rel_ist_edm_split[entry_id].append(logdata['rel_ist_edm'][entry])

        pst_jigsaw_no_recomp_split[entry_id].append(logdata['pst_jigsaw_no_recomp'][entry])
        rel_pst_jigsaw_no_recomp_split[entry_id].append(logdata['rel_pst_jigsaw_no_recomp'][entry])
        ist_jigsaw_no_recomp_split[entry_id].append(logdata['ist_jigsaw_no_recomp'][entry])
        rel_ist_jigsaw_no_recomp_split[entry_id].append(logdata['rel_ist_jigsaw_no_recomp'][entry])

        pst_recon_split[entry_id].append(logdata['pst_jigsaw'][entry])
        rel_pst_recon_split[entry_id].append(logdata['rel_pst_jigsaw'][entry])
        ist_recon_split[entry_id].append(logdata['ist_jigsaw'][entry])
        rel_ist_recon_split[entry_id].append(logdata['rel_ist_jigsaw'][entry])

        pst_hierarchical_split[entry_id].append(logdata['pst_jigsawm'][entry])
        rel_pst_hierarchical_split[entry_id].append(logdata['rel_pst_jigsawm'][entry])
        ist_hierarchical_split[entry_id].append(logdata['ist_jigsawm'][entry])
        rel_ist_hierarchical_split[entry_id].append(logdata['rel_ist_jigsawm'][entry])
    
    ## add additional data point for mean
    for entry in range(len(benchmarks_split)):
        benchmarks_split[entry].append('GMean')
        pst_edm_split[entry].append(gmean(pst_edm_split[entry]))
        ist_edm_split[entry].append(gmean(ist_edm_split[entry]))
        rel_pst_edm_split[entry].append(gmean(rel_pst_edm_split[entry]))
        rel_ist_edm_split[entry].append(gmean(rel_ist_edm_split[entry]))
        pst_jigsaw_no_recomp_split[entry].append(gmean(pst_jigsaw_no_recomp_split[entry]))
        ist_jigsaw_no_recomp_split[entry].append(gmean(ist_jigsaw_no_recomp_split[entry]))
        rel_pst_jigsaw_no_recomp_split[entry].append(gmean(rel_pst_jigsaw_no_recomp_split[entry]))
        rel_ist_jigsaw_no_recomp_split[entry].append(gmean(rel_ist_jigsaw_no_recomp_split[entry]))
        pst_recon_split[entry].append(gmean(pst_recon_split[entry]))
        ist_recon_split[entry].append(gmean(ist_recon_split[entry]))
        rel_pst_recon_split[entry].append(gmean(rel_pst_recon_split[entry]))
        rel_ist_recon_split[entry].append(gmean(rel_ist_recon_split[entry]))
        pst_hierarchical_split[entry].append(gmean(pst_hierarchical_split[entry]))
        ist_hierarchical_split[entry].append(gmean(ist_hierarchical_split[entry]))
        rel_pst_hierarchical_split[entry].append(gmean(rel_pst_hierarchical_split[entry]))
        rel_ist_hierarchical_split[entry].append(gmean(rel_ist_hierarchical_split[entry]))


    split_eval_data = {}
    split_eval_data['benchmarks'] = benchmarks_split
    split_eval_data['pst'] = {'edm': pst_edm_split, 'jigsaw_no_recomp': pst_jigsaw_no_recomp_split, 'jigsaw':pst_recon_split, 'jigsawm': pst_hierarchical_split}
    split_eval_data['ist'] = {'edm': ist_edm_split, 'jigsaw_no_recomp': ist_jigsaw_no_recomp_split, 'jigsaw':ist_recon_split, 'jigsawm': ist_hierarchical_split}
    split_eval_data['rel_pst'] = {'edm': rel_pst_edm_split, 'jigsaw_no_recomp': rel_pst_jigsaw_no_recomp_split, 'jigsaw':rel_pst_recon_split, 'jigsawm': rel_pst_hierarchical_split}
    split_eval_data['rel_ist'] = {'edm': rel_ist_edm_split, 'jigsaw_no_recomp': rel_ist_jigsaw_no_recomp_split, 'jigsaw':rel_ist_recon_split, 'jigsawm': rel_ist_hierarchical_split}


    return split_eval_data

def get_ist_table_data(benchmarks_split,ist_edm_split, rel_ist_edm_split, ist_recon_split, rel_ist_recon_split,ist_hierarchical_split, rel_ist_hierarchical_split,machines_evaluated):
    '''
    Function to compute the statistics for the IST data 
    This is used to shrink the evaluation data presented in the paper
    '''
    # get machine specific data
    true_machine_names = ['Melbourne', 'Toronto', 'Paris', 'Cambridge', 'Manhattan']
    avg_jigsaw = 0
    avg_jigsaw_m = 0
    for ibmq in range(len(machines_evaluated)):
        reqd_numbers = [min(rel_ist_edm_split[ibmq][0:len(rel_ist_recon_split[ibmq])-1]), max(rel_ist_edm_split[ibmq][0:len(rel_ist_recon_split[ibmq])-1]), rel_ist_edm_split[ibmq][-1], min(rel_ist_recon_split[ibmq][0:len(rel_ist_recon_split[ibmq])-1]),max(rel_ist_recon_split[ibmq][0:len(rel_ist_recon_split[ibmq])-1]),rel_ist_recon_split[ibmq][-1],min(rel_ist_hierarchical_split[ibmq][0:len(rel_ist_recon_split[ibmq])-1]),max(rel_ist_hierarchical_split[ibmq][0:len(rel_ist_recon_split[ibmq])-1]),rel_ist_hierarchical_split[ibmq][-1]]
        avg_jigsaw = avg_jigsaw + rel_ist_recon_split[ibmq][-1] 
        avg_jigsaw_m = avg_jigsaw_m + rel_ist_hierarchical_split[ibmq][-1] 
        printstr = '' 
        for possible_machine in true_machine_names:
            if machines_evaluated[ibmq] in possible_machine:
                printstr = printstr + possible_machine 
                break
        for i in reqd_numbers:
            printstr = printstr + ' & ' + str(i)[0:4]
        printstr = printstr + ' \\' + '\\'+ '\n \hline \n' 
        print(printstr)
        #print('Minimum True IST for this Machine EDM ' , min(ist_edm_split[ibmq]), ' JigSaw ', min(ist_recon_split[ibmq]),' JigSaw-M ', min(ist_hierarchical_split[ibmq]))
    print('Avg. from JigSaw ', avg_jigsaw/len(machines_evaluated), ' Avg. from JigSaw- M ',  avg_jigsaw_m/len(machines_evaluated))

    for ibmq in range(len(machines_evaluated)):
        print('------- Machine ------', machines_evaluated[ibmq])
        for prog in range(len(benchmarks_split[ibmq])):
            print('Prog: ', benchmarks_split[ibmq][prog], ' EDM: ', ist_edm_split[ibmq][prog], ' JigSaw ',  ist_recon_split[ibmq][prog], ' JigSaw-M ', ist_hierarchical_split[ibmq][prog])
            #print('Relative Numbers EDM: ', rel_ist_edm_split[ibmq][prog], ' JigSaw ',  rel_ist_recon_split[ibmq][prog], ' JigSaw-M ', rel_ist_hierarchical_split[ibmq][prog])

        print('\n')

## the same function can be used to print the TVD numbers, Fidelity numbers
def get_hdist_table_data(benchmarks_split,hdist_base_split,hdist_edm_split,hdist_jigsaw_split,hdist_jigsaw_m_split,machines_evaluated):
    '''
    Function to compute the statistics for the Hdist or TVD data 
    This is used to shrink the evaluation data presented in the paper
    '''
    true_machine_names = ['Melbourne', 'Toronto', 'Paris', 'Cambridge', 'Manhattan']
    avg_base,avg_jigsaw,avg_jigsaw_m = 0,0,0
    for ibmq in range(len(machines_evaluated)):
        reqd_numbers = [min(hdist_edm_split[ibmq][0:len(hdist_jigsaw_split[ibmq])-1]), max(hdist_edm_split[ibmq][0:len(hdist_jigsaw_split[ibmq])-1]), hdist_edm_split[ibmq][-1], min(hdist_jigsaw_split[ibmq][0:len(hdist_jigsaw_split[ibmq])-1]),max(hdist_jigsaw_split[ibmq][0:len(hdist_jigsaw_split[ibmq])-1]),hdist_jigsaw_split[ibmq][-1],min(hdist_jigsaw_m_split[ibmq][0:len(hdist_jigsaw_m_split[ibmq])-1]),max(hdist_jigsaw_m_split[ibmq][0:len(hdist_jigsaw_m_split[ibmq])-1]),hdist_jigsaw_m_split[ibmq][-1]] 
        avg_base = avg_base + hdist_base_split[ibmq][-1]
        avg_jigsaw = avg_jigsaw + hdist_jigsaw_split[ibmq][-1]
        avg_jigsaw_m = avg_jigsaw_m + hdist_jigsaw_m_split[ibmq][-1]
        printstr = '' 
        for possible_machine in true_machine_names:
            if machines_evaluated[ibmq] in possible_machine:
                printstr = printstr + possible_machine 
                break
        for i in reqd_numbers:
            printstr = printstr + ' & ' + str(i)[0:4]
        printstr = printstr + ' \\' + '\\'+ '\n \hline \n' 
        print(printstr)
        #print('Minimum True IST for this Machine EDM ' , min(ist_edm_split[ibmq]), ' JigSaw ', min(ist_recon_split[ibmq]),' JigSaw-M ', min(ist_hierarchical_split[ibmq]))
    print('Avg. from Baseline ', avg_base/len(machines_evaluated), 'Avg. from JigSaw ', avg_jigsaw/len(machines_evaluated), ' Avg. from JigSaw- M ',  avg_jigsaw_m/len(machines_evaluated))
    print('Relative numbers: Jigsaw wrt Baseline:- ', (avg_jigsaw/len(machines_evaluated))/(avg_base/len(machines_evaluated)))
    print('Relative numbers: Jigsaw-M wrt Baseline:- ', (avg_jigsaw_m/len(machines_evaluated))/(avg_base/len(machines_evaluated)))

    for ibmq in range(len(machines_evaluated)):
        print('------- Machine ------', machines_evaluated[ibmq])
        for prog in range(len(benchmarks_split[ibmq])):
            print('Prog: ', benchmarks_split[ibmq][prog], ' EDM: ', hdist_edm_split[ibmq][prog], ' JigSaw ',  hdist_jigsaw_split[ibmq][prog], ' JigSaw-M ', hdist_jigsaw_m_split[ibmq][prog])
            #print('Relative Numbers EDM: ', rel_ist_edm_split[ibmq][prog], ' JigSaw ',  rel_ist_recon_split[ibmq][prog], ' JigSaw-M ', rel_ist_hierarchical_split[ibmq][prog])

        print('\n')


def plot_data_three_machines_only(plot_name, benchmarks_split, pst_edm_split, rel_pst_edm_split, ist_edm_split, rel_ist_edm_split,
                pst_recon_split, rel_pst_recon_split, ist_recon_split, rel_ist_recon_split,
                pst_hierarchical_split, rel_pst_hierarchical_split, ist_hierarchical_split, rel_ist_hierarchical_split,
                plot_type,plot_data_type,fig_dim,color,delta,label_dist,adjust_size,fonts,rotation_value,legend_opts,machines_evaluated):
    m0_size = int(len(rel_pst_edm_split[0])) #mel 
    m1_size = int(len(rel_pst_edm_split[1])) #tor 
    m2_size = int(len(rel_pst_edm_split[2])) #par 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,gridspec_kw={'width_ratios': [m0_size,m1_size,m2_size]})
    fig.set_size_inches(fig_dim[0], fig_dim[1])
    if(plot_type == 'Hierarchical'):
        ind_ax1 = 2.0*np.arange(m0_size)  # the x locations for the groups
        ind_ax2 = 2.0*np.arange(m1_size)  # the x locations for the groups
        ind_ax3 = 2.0*np.arange(m2_size)  # the x locations for the groups
    else:
        ind_ax1 = 1.5*np.arange(m0_size)  # the x locations for the groups
        ind_ax2 = 1.5*np.arange(m1_size)  # the x locations for the groups
        ind_ax3 = 1.5*np.arange(m2_size)  # the x locations for the groups
    width = 0.4       # the width of the bars
    if(plot_data_type == 'PST'):
        rects1 = ax1.bar(ind_ax1, rel_pst_edm_split[0], width, color=color[0],edgecolor='black')
        rects2 = ax2.bar(ind_ax2, rel_pst_edm_split[1], width, color=color[0],edgecolor='black')
        rects3 = ax3.bar(ind_ax3, rel_pst_edm_split[2], width, color=color[0],edgecolor='black')

        rects5 = ax1.bar(ind_ax1 + width, rel_pst_recon_split[0], width, color=color[1],edgecolor='black')
        rects6 = ax2.bar(ind_ax2 + width, rel_pst_recon_split[1], width, color=color[1],edgecolor='black')
        rects7 = ax3.bar(ind_ax3 + width, rel_pst_recon_split[2], width, color=color[1],edgecolor='black')

        rects9 = ax1.bar(ind_ax1 + 2*width, rel_pst_hierarchical_split[0], width, color=color[2],edgecolor='black')
        rects10 = ax2.bar(ind_ax2 + 2*width, rel_pst_hierarchical_split[1], width, color=color[2],edgecolor='black')
        rects11 = ax3.bar(ind_ax3 + 2*width, rel_pst_hierarchical_split[2], width, color=color[2],edgecolor='black')
    relative_loc = width/2
    relative_loc = width
    x = ind_ax1 + relative_loc
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks_split[0],fontsize=fonts[0], rotation=rotation_value)
    x = ind_ax2 + relative_loc
    ax2.set_xticks(x)
    ax2.set_xticklabels(benchmarks_split[1],fontsize=fonts[0], rotation=rotation_value)
    x = ind_ax3 + relative_loc
    ax3.set_xticks(x)
    ax3.set_xticklabels(benchmarks_split[2],fontsize=fonts[0], rotation=rotation_value)

    ### PD: Enforcing yticks for toronto paris and manhattan
    yticks_locs= [2,4,6,8]
    ax1.set_yticks(yticks_locs)
    ax2.set_yticks(yticks_locs)
    ax3.set_yticks(yticks_locs)
    ax1.set_yticklabels(yticks_locs,fontsize=10)
    ax2.set_yticklabels(yticks_locs,fontsize=10)
    ax3.set_yticklabels(yticks_locs,fontsize=10)
    ax1.axhline(y=1, color='#555555', linestyle='--')
    ax2.axhline(y=1, color='#555555', linestyle='--')
    ax3.axhline(y=1, color='#555555', linestyle='--')

    ax1.set_ylabel('Relative Probability of\nSuccessful Trial (PST)',fontsize=fonts[1])

    true_machine_names = ['Melbourne (15-qubits)', 'IBMQ-Toronto (27-qubits)', 'IBMQ-Paris (27-qubits)', 'IBMQ-Cambridge (28-qubits)', 'IBMQ-Manhattan (65-qubits)']
    machine_acronyms = ['Mel', 'Tor', 'Par','Cam','Man']
    xlabels = [] 
    for i in range(len(machines_evaluated)): # four machines are evaluated 
        for j in range(len(machine_acronyms)):
            if(machines_evaluated[i]==machine_acronyms[j]):
                xlabels.append(true_machine_names[j])
                break
    #ax2.set_xlabel('Benchmarks (Machine)',fontsize=fonts[2])
    ax1.set_xlabel(xlabels[0],fontsize=fonts[2])
    ax2.set_xlabel(xlabels[1],fontsize=fonts[2])
    ax3.set_xlabel(xlabels[2],fontsize=fonts[2])
    #plt.xlabel('Benchmarks (Machine)',fontsize=fonts[2])
    ylim_m0_val = 0
    ylim_m1_val = 0
    ylim_m2_val = 0
    ylim_m0_val= max(max(rel_pst_edm_split[0]), max(rel_pst_recon_split[0]), max(rel_pst_hierarchical_split[0])) + delta[0]
    ylim_m1_val = max(max(rel_pst_edm_split[1]), max(rel_pst_recon_split[1]), max(rel_pst_hierarchical_split[1])) + delta[1]
    ylim_m2_val = max(max(rel_pst_edm_split[2]), max(rel_pst_recon_split[2]), max(rel_pst_hierarchical_split[2])) + delta[2]

    ax1.set_ylim(0,ylim_m0_val)
    ax2.set_ylim(0,ylim_m1_val)
    ax3.set_ylim(0,ylim_m2_val)

    plt.rcParams['ytick.labelsize'] = 8
    yticks = ax1.yaxis.get_major_ticks()
    #yticks[0].label1.set_visible(False)
    yticks = ax2.yaxis.get_major_ticks()
    #yticks[0].label1.set_visible(False)
    yticks = ax3.yaxis.get_major_ticks()
    #yticks[0].label1.set_visible(False)
    plt.subplots_adjust(top=adjust_size)
    fig.legend((rects1[0], rects5[0],rects9[0]), ('EDM', 'JigSaw','JigSaw-M'),loc='upper center',ncol=legend_opts[0],fontsize= legend_opts[1])
    def autolabel(rects,array,axis,dist):
        """  
        Attach a text label above each bar displaying its height
        """
        ctr = 0
        label_array = [EM.truncate(v*100,1) for v in array]
        for entry in range(len(label_array)):
            if(label_array[entry]>=0) and (label_array[entry]<=1):
                label_array[entry] = EM.truncate(array[entry]*100,2)


        for rect in rects:
            height = rect.get_height()
            if(axis=='1'):
                ax1.text(rect.get_x() + rect.get_width()/2., height+dist,
                    label_array[ctr],fontsize=fonts[3],
                    #'%d' % int(height),
                    ha='center', va='bottom',rotation=90)
            elif(axis=='2'):
                ax2.text(rect.get_x() + rect.get_width()/2., height+dist,
                    label_array[ctr],fontsize=fonts[3],
                    #'%d' % int(height),
                    ha='center', va='bottom',rotation=90)
            elif(axis=='3'):
                ax3.text(rect.get_x() + rect.get_width()/2., height+dist,
                    label_array[ctr],fontsize=fonts[3],
                    #'%d' % int(height),
                    ha='center', va='bottom',rotation=90)
            elif(axis=='4'):
                ax4.text(rect.get_x() + rect.get_width()/2., height+dist,
                    label_array[ctr],fontsize=fonts[3],
                    #'%d' % int(height),
                    ha='center', va='bottom',rotation=90)
            ctr = ctr + 1

    if(plot_data_type == 'PST' and plot_type != 'Hierarchical'):
        autolabel(rects1,pst_edm_split[0],'1',label_dist[0])
        autolabel(rects2,pst_edm_split[1],'2',label_dist[1])
        autolabel(rects3,pst_edm_split[2],'3',label_dist[2])
        autolabel(rects4,pst_edm_split[3],'4',label_dist[3])
        autolabel(rects5,pst_recon_split[0],'1',label_dist[0])
        autolabel(rects6,pst_recon_split[1],'2',label_dist[1])
        autolabel(rects7,pst_recon_split[2],'3',label_dist[2])
        autolabel(rects8,pst_recon_split[3],'4',label_dist[3])
    plt.savefig(plot_name, bbox_inches="tight")
    plot_show_on = 1
    display_grayscale = 0
    if(display_grayscale):
        bnw_image_name = plot_name.replace('pdf', 'jpg')
        plt.savefig(bnw_image_name, bbox_inches="tight")
    if(plot_show_on):
        plt.show()
    if(display_grayscale):
        img=Image.open(bnw_image_name).convert('L')
        img.save(bnw_image_name)
        im=mpimg.imread(bnw_image_name)
        a=plt.imshow(im, cmap = "gray")
        if(int(plot_show_on)==1):
            plt.show(a)


