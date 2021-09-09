from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import helper_functions as helper
import Reconstruction_Functions as RF
import Eval_Metrics as EM
import ast
import get_data_from_logfile as gdata
from scipy.stats.mstats import gmean

from PIL import Image
import matplotlib.image as mpimg
import statistics
import random
from random import randint
import time
import tracemalloc
import helper_reconstruction_graphs as RG

existing_filename = './isca_jigsaw_data.txt'
machines_evaluated = ['Tor', 'Par', 'Man']
true_machine_names = ['Melbourne (15-qubits)', 'Toronto (27-qubits)', 'Paris (27-qubits)', 'Cambridge (28-qubits)', 'Manhattan (65-qubits)']

reconstruct_and_plot = 0
table_data = 1
stats = 0
ploton = 0 + reconstruct_and_plot
mean_plot =0

## List of logfiles
logdir = '../logfiles/'
logfiles = gdata.get_all_experimental_data('./isca_final_evaluations.txt')

## Parse out the information
logfilelist = []
xlabels = []
num_solutions = []
for ele in logfiles:
    logfile_vector = []
    for idx in range(2,len(ele)):
        logfile_vector.append(logdir+ele[idx])
    
    logfilelist.append(logfile_vector)
    xlabels.append(ele[0])
    num_solutions.append(int(ele[1]))

print('Gathered all information')

if(int(reconstruct_and_plot)==1):
    print('Beginning Reconstruction')
    logdata = RG.perform_reconstruction(benchmarks=xlabels,logfile_list= logfilelist,num_solutions= num_solutions,outputfilename=existing_filename)
else:
    print('Reading Reconstruction Data from logfile ')
    logdata = RG.get_reconstruction_data_from_outputlogfile(existing_filename)
    
if(int(stats)==1):
    print('Generating Statistics ')
    RG.generate_statistics(machines_evaluated,logdata)

## separate out data for diff machines
print('Obtaining Machine specific data from reconstruction outputs')
eval_metrics = RG.get_machine_specific_data(machines_evaluated,logdata)

dist_metrics = RG.get_machine_specific_distribution_metrics_data(machines_evaluated,logdata) 

## params
labelfontsize = 12
fig_dim = np.array([10.0, 2.00])

## gist
Ncolors = 4
color_palette = sns.color_palette("gist_earth_r",Ncolors)

print('Plotting data ')
plot_name= "../plots/jigsaw_relative_pst_real_machine_data.pdf"
if(int(ploton) ==1):
    fig_dim = np.array([10.0, 2.0])
    RG.plot_data_three_machines_only(plot_name, eval_metrics['benchmarks'], eval_metrics['pst']['edm'], eval_metrics['rel_pst']['edm'], eval_metrics['ist']['edm'], eval_metrics['rel_ist']['edm'],
                eval_metrics['pst']['jigsaw'], eval_metrics['rel_pst']['jigsaw'], eval_metrics['ist']['jigsaw'], eval_metrics['rel_ist']['jigsaw'], eval_metrics['pst']['jigsawm'], eval_metrics['rel_pst']['jigsawm'], eval_metrics['ist']['jigsawm'], eval_metrics['rel_ist']['jigsawm'],
                'Hierarchical','PST',fig_dim,color_palette,[0.2,0.2,0.2,0.2],[0.05,0.15,0.25,0.05],0.85,[7,10,8,7],
                75,[3,'small'],machines_evaluated)

## Have a separate plot to show impact with or without recompilation

if int(mean_plot)==1:
    xlabel = ['IBMQ-Toronto', 'IBMQ-Paris', 'IBMQ-Manhattan']
    plot_name= "../plots/jigsaw_subsetting_vs_recompilation_machine_data.pdf"
    avg_fidelity = np.zeros((4,len(machines_evaluated)))
    for i in range(len(machines_evaluated)):
        avg_fidelity[0][i] = rel_pst_edm_split[i][-1]
        avg_fidelity[1][i] = rel_pst_jigsaw_no_recomp_split[i][-1]
        avg_fidelity[2][i] = rel_pst_recon_split[i][-1]
        avg_fidelity[3][i] = rel_pst_hierarchical_split[i][-1]
    PL.dataplot(y_axis= avg_fidelity,plot_name = plot_name,plot_type='bar',ylabel='Mean Relative PST',xticks_labels=xlabel,palette_style='gist_r',legends=['EDM','JigSaw w/o Recompilation','JigSaw with Recompilation','JigSaw-M with Recompilation'],figure_dim = [8,2.2],xticks_rotation=0,xticks_fontsize=12,plot_baseline=1,plot_baseline_color='#555555',yticks_locs = [1,2,3,4], yticks_labels= [1,2,3,4],legend_size=12,ylabel_size=12,yticks_fontsize=12,ylim=[0,6])

    
if int(table_data):
    RG.get_ist_table_data(eval_metrics['benchmarks'], eval_metrics['ist']['edm'], eval_metrics['rel_ist']['edm'], eval_metrics['ist']['jigsaw'], eval_metrics['rel_ist']['jigsaw'],eval_metrics['ist']['jigsawm'], eval_metrics['rel_ist']['jigsawm'],machines_evaluated)
    print('Hellinger Distance Data')
    RG.get_hdist_table_data(dist_metrics['benchmarks'],dist_metrics['hdist']['base'],dist_metrics['hdist']['edm'],dist_metrics['hdist']['jigsaw'],dist_metrics['hdist']['jigsawm'],machines_evaluated)
    print('Printing Relative Data for Correlation')
    RG.get_ist_table_data(dist_metrics['benchmarks'], dist_metrics['corr']['edm'], dist_metrics['rel_corr']['edm'], dist_metrics['corr']['jigsaw'], dist_metrics['rel_corr']['jigsaw'], dist_metrics['corr']['jigsawm'], dist_metrics['rel_corr']['jigsawm'],machines_evaluated)
    print('Total Variation Distance Data')
    RG.get_hdist_table_data(dist_metrics['benchmarks'],dist_metrics['tvd']['base'],dist_metrics['tvd']['edm'],dist_metrics['tvd']['jigsaw'],dist_metrics['tvd']['jigsawm'],machines_evaluated)
    print('Printing Relative Data for Fidelity')
    RG.get_ist_table_data(dist_metrics['benchmarks'], dist_metrics['fidelity']['edm'], dist_metrics['rel_fidelity']['edm'], dist_metrics['fidelity']['jigsaw'], dist_metrics['rel_fidelity']['jigsaw'], dist_metrics['fidelity']['jigsawm'], dist_metrics['rel_fidelity']['jigsawm'],machines_evaluated)

