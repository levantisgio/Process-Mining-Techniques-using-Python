'''
Γιώργος Λεβαντής
mpked2216
29/01/2023

Δυσκολεύτηκα αρκετά ώστε να βρώ τις εκδόσεις και τις καινούριες εντολές ώστε να τρέξουν όλα.
Οι εκδόσεις πάνω στις οποίες τρέχει χωρίς πρόβλημα ο κώδικας ειναι οι εξής:

PYTHON 3.9
PM4PY 2.5.0
'''

import datetime as dt
import sys

import pandas as pd
import pm4py

from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.log import converter as log_conv_fact
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.heuristics_net import visualizer as pn_vis
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator


# step 1: read the event log

log = pm4py.read_xes("/Users/levantisgio/Desktop/BPI_Challenge_2017.xes")
print(log)
event_log = log_conv_fact.apply(log, variant=log_conv_fact.TO_EVENT_STREAM)


#step 2
event_attributes = pm4py.get_event_attributes(log)
trace_attributes = pm4py.get_trace_attributes(log)
print('Event\'s structure:', event_attributes)
print('\n\n\n')
print('Trace\'s structure:',trace_attributes)

#print("Trace structure:", log.keys())
#print("Event structure:", log[0][0].keys()) # Den trexei..

# step 3: display the number of traces

print('Number of traces: ', log['case:concept:name'].nunique())

# step 4: display the number of events

print("Number of events: ", len(event_log))

# step 5: Show different events of Event Log (Based on event's concept:name)

print("5. Show different events of Event Log (Based on event's concept:name)")
print("=====================================================================")
dif_events = []
for event in event_log:
    flag=0
    for dif in dif_events:
        if dif == event['concept:name']:
            flag=1
            break

    if flag!=1:
        print(event['concept:name'])
        dif_events.append(event['concept:name'])

print('===================================')
print('\n\n\n')

# step 6: Build a filtered event log

filtered = pm4py.filter_time_range(log, dt.datetime(2017, 1, 6), dt.datetime(2017, 1, 7), mode='events')
event_filtered = log_conv_fact.apply(filtered, variant=log_conv_fact.TO_EVENT_STREAM)
print(filtered)

event_attributes_filtered = pm4py.get_event_attributes(filtered)
trace_attributes_filtered = pm4py.get_trace_attributes(filtered)
print('Event\'s structure:', event_attributes_filtered)
print('\n\n\n')
print('Trace\'s structure:',trace_attributes_filtered)

print('Number of traces: ', filtered['case:concept:name'].nunique())
print("Number of events: ", len(event_log))

'''
#num_events_f = 0
#for trace in filtered:
#    num_events_f += len(trace)
#print("Number of events:", num_events_f)
'''


 #ALPHA MINER


#LOG

def execute_alpha():
    net, i_m, f_m = alpha_miner.apply(log)
    gviz= pn_vis.apply(net, i_m, f_m)
    #pn_vis.save(gviz, "alphaminer.jpg")
    pn_vis.view(gviz)
    #print(dir(net))
    #print(net.arcs)
    #print(net.transitions)
    #print(net.places)
execute_alpha()

'''
#Fitness, Precesion, Genetalization, Simplicity (LOG)
'''
net, i_m, f_m = alpha_miner.apply(log)

fitness = pm4py.fitness_token_based_replay(log, net, i_m, f_m)
print("FITNESS", fitness)

prec = pm4py.precision_token_based_replay(log, net, i_m, f_m)
print("PRECISION", prec)

gen = generalization_evaluator.apply(log, net, i_m, f_m)
print("GENERALIZATION", gen)

simp = simplicity_evaluator.apply(net)
print("SIMPLICITY", simp)

# FILTERED

def execute_alpha_filtered():
    net, i_m, f_m = alpha_miner.apply(filtered)
    gviz= pn_vis.apply(net, i_m, f_m)
    #pn_vis.save(gviz, "alphaminer_filtered.jpg")
    pn_vis.view(gviz)
    #print(dir(net))
    #print(net.arcs)
    #print(net.transitions)
    #print(net.places)

execute_alpha_filtered()


'''
#Fitness, Precesion, Genetalization, Simplicity (FILTERED)
'''
net, i_m, f_m = alpha_miner.apply(filtered)
fitness = pm4py.fitness_token_based_replay(filtered, net, i_m, f_m)
print("FITNESS", fitness)

prec = pm4py.precision_token_based_replay(filtered, net, i_m, f_m)
print("PRECISION", prec)

gen = generalization_evaluator.apply(filtered, net, i_m, f_m)
print("GENERALIZATION", gen)

simp = simplicity_evaluator.apply(net)
print("SIMPLICITY", simp)




 #HEURISTICS


# LOG

str1 = "/Users/levantisgio/opt/anaconda3/lib/python3.9/site-packages/graphviz/"
sys.path.append(str1)

def execute_heuristics():
    net, i_m, f_m = heuristics_miner.apply(log)
    gviz= pn_vis.apply(net, i_m, f_m)
    #pn_vis.save(gviz, "heuristics.jpg")
    pn_vis.view(gviz)
    #print(dir(net))
    #print(net.arcs)
    #print(net.transitions)
    #print(net.places)

execute_heuristics()


'''
#Fitness, Precesion, Genetalization, Simplicity (LOG)
'''

net, i_m, f_m = heuristics_miner.apply(log)

fitness_h_log = pm4py.fitness_token_based_replay(log, net, i_m, f_m)
print("FITNESS", fitness_h_log)

prec_h_log = pm4py.precision_token_based_replay(log, net, i_m, f_m)
print("PRECISION", prec_h_log)

gen_h_log = generalization_evaluator.apply(log, net, i_m, f_m)
print("GENERALIZATION", gen_h_log)

simp_h_log = simplicity_evaluator.apply(net)
print("SIMPLICITY", simp_h_log)


str1 = "/library/frameworks/python.framework/versions/3.6/lib/python3.6/site-packages/graphviz/"
sys.path.append(str1)

def execute_heuristics():
    net = heuristics_miner.apply_heu(log)
    gviz= pn_vis.apply(net)
    #pn_vis.save(gviz, "heuristics_heu.jpg")
    pn_vis.view(gviz)
execute_heuristics()


#FILTERED

str1 = "/Users/levantisgio/opt/anaconda3/lib/python3.9/site-packages/graphviz/"
sys.path.append(str1)

def execute_heuristics_filtered():
    net, i_m, f_m = heuristics_miner.apply(filtered)
    gviz= pn_vis.apply(net, i_m, f_m)
    #pn_vis.save(gviz, "heuristics_filtered.jpg")
    pn_vis.view(gviz)
    #print(dir(net))
    #print(net.arcs)
    #print(net.transitions)
    #print(net.places)

execute_heuristics_filtered()

'''
#Fitness, Precesion, Genetalization, Simplicity (FILTERED)
'''
net, i_m, f_m = heuristics_miner.apply(filtered)

fitness_h_filtered = pm4py.fitness_token_based_replay(filtered, net, i_m, f_m)
print("FITNESS", fitness_h_filtered)

prec_h_filtered = pm4py.precision_token_based_replay(filtered, net, i_m, f_m)
print("PRECISION", prec_h_filtered)

gen_h_filtered= generalization_evaluator.apply(filtered, net, i_m, f_m)
print("GENERALIZATION", gen_h_filtered)

simp_h_filtered = simplicity_evaluator.apply(net)
print("SIMPLICITY", simp_h_filtered)

'''
#str1 = "/library/frameworks/python.framework/versions/3.6/lib/python3.6/site-packages/graphviz/"
#sys.path.append(str1)
#
#def execute_heuristics_filtered():
#
#    net = heuristics_miner.apply_heu(filtered)
#    gviz= pn_vis.apply(net)
#    #pn_vis.save(gviz, "heuristics_heu_filtered.jpg")
#    pn_vis.view(gviz)
#
#execute_heuristics_filtered()
'''




#INDUCTIVE


# LOG

def execute_inductive():
    pt = inductive_miner.apply(log)
    net, i_m, f_m = pt_converter.apply(pt)
    gviz = pn_vis.apply(net, i_m, f_m)
    #pn_vis.save(gviz,"inductive.jpg")
    pn_vis.view(gviz)
    #print(dir(net))
    #print(net.arcs)
    #print(net.transitions)
    #print(net.places)

execute_inductive()

'''
#Fitness, Precesion, Genetalization, Simplicity (LOG)
'''
pt = inductive_miner.apply(log)
net, i_m, f_m = pt_converter.apply(pt)

fitness_ind_log = pm4py.fitness_token_based_replay(log, net, i_m, f_m)
print("FITNESS", fitness_ind_log)

prec_ind_log = pm4py.precision_token_based_replay(log, net, i_m, f_m)
print("PRECISION", prec_ind_log)

gen_ind_log = generalization_evaluator.apply(log, net, i_m, f_m)
print("GENERALIZATION", gen_ind_log)

simp_ind_log = simplicity_evaluator.apply(net)
print("SIMPLICITY", simp_ind_log)


'''
#from pm4py.algo.discovery.inductive import algorithm as inductive_miner
#net, i_m, f_m = pm4py.discover_petri_net_inductive(log)
#tree = pm4py.discover_process_tree_inductive(log)
#pm4py.view_process_tree(tree)
#net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
'''


# FILTERED

def execute_inductive_filtered():
    pt = inductive_miner.apply(filtered)
    net, i_m, f_m = pt_converter.apply(pt)
    gviz = pn_vis.apply(net, i_m, f_m)
    #pn_vis.save(gviz,"inductive_filtered.jpg")
    pn_vis.view(gviz)
    #print(dir(net))
    #print(net.arcs)
    #print(net.transitions)
    #print(net.places)

execute_inductive_filtered()


#Fitness, Precesion, Genetalization, Simplicity (FILTERED)


pt = inductive_miner.apply(filtered)
net, i_m, f_m = pt_converter.apply(pt)

fitness_ind_filtered = pm4py.fitness_token_based_replay(filtered, net, i_m, f_m)
print("FITNESS", fitness_ind_filtered)

prec_ind_filtered = pm4py.precision_token_based_replay(filtered, net, i_m, f_m)
print("PRECISION", prec_ind_filtered)

gen_ind_filtered = generalization_evaluator.apply(filtered, net, i_m, f_m)
print("GENERALIZATION", gen_ind_filtered)

simp_ind_filtered = simplicity_evaluator.apply(net)
print("SIMPLICITY", simp_ind_filtered)
'''

'''
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
net, i_m, f_m = pm4py.discover_petri_net_inductive(filtered)
tree = pm4py.discover_process_tree_inductive(filtered)
pm4py.view_process_tree(tree)
net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)



# ------------ STEP 8 -------------------

# FITNESS

#fitness = pm4py.fitness_token_based_replay(log, net, i_m, f_m)
#print("FITNESS", fitness)
#fitness = pm4py.fitness_alignments(log, net, i_m, f_m)
#print("FITNESS", fitness)


# PRECISION

#prec = pm4py.precision_token_based_replay(log, net, i_m, f_m)
#print("PRECISION", prec)
#prec = pm4py.precision_alignments(log, net, i_m, f_m)
#print("PRECISION", prec)


# GENERALIZATION

#gen = generalization_evaluator.apply(log, net, i_m, f_m)
#print("GENERALIZATION", gen)


# SIMPLICITY

#simp = simplicity_evaluator.apply(net)
#print("SIMPLICITY", simp)


# ----------------- STEP 9 ---------------------

 #             REPLAY FITNESS

# 1. ALPHA MINER

net_a_log, i_m_a_log, f_m_a_log = alpha_miner.apply(log)
replayed_traces_original_log_alpha = pm4py.conformance_diagnostics_token_based_replay(log,net_a_log,i_m_a_log,f_m_a_log)
#print(replayed_traces_original_log_heuristics)
count_not_fit_traces_alpha = 0
for trace in replayed_traces_original_log_alpha:
    if trace['trace_is_fit'] ==False:
        count_not_fit_traces_alpha +=1

print('Number of not fitted Traces using Aalpha Miner Algorithm:', count_not_fit_traces_alpha)
print('% of not fitted Traces using Alpha Miner Algorithm:',(count_not_fit_traces_alpha/log['case:concept:name'].nunique())*100)

# 2. HEURISTICS
net_h_log,i_m_h_log,f_m_h_log = heuristics_miner.apply(log)
replayed_traces_original_log_heuristics = pm4py.conformance_diagnostics_token_based_replay(log,net_h_log,i_m_h_log,f_m_h_log)
#print(replayed_traces_original_log_heuristics)
count_not_fit_traces_heuristics = 0
for trace in replayed_traces_original_log_heuristics:
    if trace['trace_is_fit'] ==False:
        count_not_fit_traces_heuristics +=1

print('Number of not fitted Traces using Heuristics Algorithm:', count_not_fit_traces_heuristics)
print('% of not fitted Traces using Heuristics Algorithm:',(count_not_fit_traces_heuristics/log['case:concept:name'].nunique())*100)

# 3. INDUCTIVE

pt = inductive_miner.apply(log)
net_i_log,i_m_i_log,f_m_i_log = pt_converter.apply(pt)
replayed_traces_original_log_inductive = pm4py.conformance_diagnostics_token_based_replay(log,net_i_log,i_m_i_log,f_m_i_log)
#print(replayed_traces_original_log_heuristics)
count_not_fit_traces_inductive = 0
for trace in replayed_traces_original_log_inductive:
    if trace['trace_is_fit'] == False:
        count_not_fit_traces_inductive +=1

print('Number of not fitted Traces using Inductive Algorithm:', count_not_fit_traces_inductive)
print('% of not fitted Traces using Inductive Algorithm:',(count_not_fit_traces_inductive/log['case:concept:name'].nunique())*100)




'''

ALIGNMENTS

ALPHA MINER
net, i_m, f_m = alpha_miner.apply(log)

HEURISTICS
net, i_m, f_m = heuristics_miner.apply(log)

INDUCTIVE
pt = inductive_miner.apply(log)
net, i_m, f_m = pt_converter.apply(pt)

aligned_traces = alignments.apply_log(log, net, i_m, f_m)
print(aligned_traces)

'''


# RESULTS


#LOG
data_log = {'Alpha Miner': [0.42, 0.06, 0.98, 0.79],
            'Heuristics': [0.95, 0.87, 0.91, 0.51],
            'Inductive': [1, 0.19, 0.94, 0.62]}

df_log = pd.DataFrame(data_log, index=['Fitness', 'Precision', 'Generalization', 'Simplicity'])
print(df_log)

#FILTERED

data_filtered = {'Alpha Miner': [0.35, 0.40, 0.72, 1.0],
        'Heuristics': [0.94, 0.93, 0.60, 0.53],
       'Inductive':[1.0, 0.48, 0.79, 0.63]}

df_filtered = pd.DataFrame(data_filtered, index=['Fitness', 'Precision', 'Generalization', 'Simplicity'])
print(df_filtered)
