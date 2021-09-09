from collections import Counter
import Reconstruction_Functions as RF
import numpy as np
import math

def compute_pst(ideal_histogram,noisy_histogram,num_sols):
    ''' 
    Function to compute PST
    Input: ideal_histogram -> output from the ideal simulator
           noisy_histogram -> output from the actual device
           num_sols -> number of solutions to be considered for computing the PST (for example 1 for BV and 2 for GHZ)
    '''
    # determine the total best solutions from ideal
    sorted_histogram = sorted(ideal_histogram.items(), key=lambda x: x[1], reverse=True)
    successful_trials_counter = 0 
    for i in range(num_sols):
        search_key = sorted_histogram[i][0]
        for key,value in noisy_histogram.items():
            if(key == search_key):
                successful_trials_counter = successful_trials_counter + value
    # compute PST
    total_trials = sum(noisy_histogram.values())
    if(successful_trials_counter <=1.0): #already a pdf
        pst = successful_trials_counter
    else:
        pst = successful_trials_counter/total_trials
    return pst 

def compute_ist(ideal_histogram,noisy_histogram,num_sols):
    ''' 
    Function to compute IST
    Input: ideal_histogram -> output from the ideal simulator
           noisy_histogram -> output from the actual device
           num_sols -> number of solutions to be considered for computing the PST (for example 1 for BV and 2 for GHZ)
    '''
    # determine the total best solutions from ideal
    sorted_histogram = sorted(ideal_histogram.items(), key=lambda x: x[1], reverse=True)
    # sort the noisy histogram
    sorted_noisy_histogram = sorted(noisy_histogram.items(), key=lambda x: x[1], reverse=True)
    # probability of correct answer
    successful_trials_counter = 0 
    for i in range(num_sols):
        search_key = sorted_histogram[i][0]
        for key,value in noisy_histogram.items():
            if(key == search_key):
                successful_trials_counter = successful_trials_counter + value
    # get the solution keys
    solution_keys = []
    for j in range(num_sols):
        solution_keys.append(sorted_histogram[j][0])
        
    error_counter = 0 
    for i in range(len(sorted_noisy_histogram)):
        search_key = sorted_noisy_histogram[i][0]
        if search_key not in solution_keys:
            error_counter = sorted_noisy_histogram[i][1]
            break
    ist = successful_trials_counter/error_counter

    return ist 



def update_dist(dict1,dict2):
    ''' 
    Function to merge two dictionaries in to a third one
    Input : dict1, dict2: two dictionaries that must be combined
    Output: merged dictionary from both
    '''
    dict3 = Counter(dict1) + Counter(dict2) 
    dict3 = dict(dict3)
    return dict3

def truncate(number, decimals=0):
    """ 
    Function that returns a value truncated to a specific number of decimal places.
    (useful for plots etc)
    Input  : number   -> that needs to be truncated
             decimal  -> how many bits to keep after decimal post truncation
    Output : truncated decimal number
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor
    
def removekey(d, key_list):
    '''
    Function that removes a key from a list of keys stored in a dictionary
    Input  : d        -> dictionary 
             key_list -> list of keys that should be removed
    Output : modified dictionary after keys removal (must be normalized after this) 
    '''
    for i in key_list:
        r = dict(d)
        del r[i]
    
    return r
    
    
def Compute_PST(dict_in, correct_answer):
    '''
    Function that computes the PST using a list of correct answers (prepared from the ideal output distribution)
    Input  : dict_in  -> output dictionary from execution
             correct_answer -> list of correct answers obtained by running the ideal simulation
    Output : PST (Prob. of Successful Trial)
    ''' 
    _in = dict_in.copy()
    norm_dict=RF.normalize_dict(_in)
    output=0
    for ele in correct_answer:
        output+=norm_dict[ele] 
    pst = output/len(correct_answer)
    
    return pst

    
def Compute_IST(dict_in, correct_answer):
    '''
    Function that computes the PST using a list of correct answers (prepared from the ideal output distribution)
    Input  : dict_in  -> output dictionary from execution
             correct_answer -> list of correct answers obtained by running the ideal simulation
    Output : PST (Prob. of Successful Trial)
    ''' 
    
    #delete correct answers from the input dict 
    # DON NOT renormalize 
    norm_dict=RF.normalize_dict(dict_in)
    pst = Compute_PST(norm_dict, correct_answer)
    test_in=removekey(norm_dict, correct_answer)
    dominant_Incorr=Counter(test_in).most_common(1)[0][1]         
    
    return pst/dominant_Incorr
    
    
def Compute_KL_Ideal(dict_in,dict_ideal):
    '''
    Function that computes the KL divergence between two distributions (preferably the ideal and noisy distribution)
    Input  : dict_in    -> output dictionary from execution
             dict_ideal -> output dictionary from the ideal simulations (but other reference distribution may be used too) 
    Output : KL divergence 
    ''' 
    
    epsilon = 0.000001
    _in1 = Counter(dict_in.copy())
    _in2 = Counter(dict_ideal.copy())
    a = Counter(dict.fromkeys(dict_in, epsilon))
    b = Counter(dict.fromkeys(dict_ideal, epsilon))
    
    P = list(dict(Counter(_in1) + Counter(b)).values())
    Q = list(dict(Counter(_in2) + Counter(a)).values())
    
    P = np.asarray(P, dtype=np.float)
    Q = np.asarray(Q, dtype=np.float)
    #print(kl_divergence(P,Q))
    # You may want to instead make copies to avoid changing the np arrays
    
    return sum(P*np.log(P/Q))

def kl_divergence(p, q):
    '''
    Function to compute the KL divergence for two probability distributions p and q
    Instead of using dictionaries, we use pdfs here
    '''
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def Compute_Entropy(dict_in):
    '''
    Function to compute the entropy of a output distribution
    Input   : dict_in  -> output distribution in the form of a dictionary
    Output  : entropy of the distribution (higher typically means more noise) 
    '''
    
    _in1 = Counter(dict_in.copy())
    epsilon = 0.000001
    P = list(dict(Counter(_in1)).values())
    a = Counter(dict.fromkeys(dict_in, epsilon))
    P = list(dict(Counter(_in1) + Counter(a)).values())
    P = np.asarray(P, dtype=np.float)
        
    return -1*sum(P*np.log(P))



def Compute_Helinger(dict_in,dict_ideal):
    '''
    Function to compute the Hellinger distance between two dictionaries
    Input  : dict_in    -> input dictionary whose Hellinger Distance must be computed
             dict_ideal -> reference or ideal dictionary against which Hellinger distance must be computed
    Output : Hellinger Distance
    ''' 
    epsilon = 0.000001
    _in1 = Counter(dict_in.copy())
    _in2 = Counter(dict_ideal.copy())
    a = Counter(dict.fromkeys(dict_in, epsilon))
    b = Counter(dict.fromkeys(dict_ideal, epsilon))

    p = list(dict(Counter(_in1) + Counter(b)).values())
    q = list((Counter(_in2) + Counter(a)).values())

    list_of_squares = []
    for p_i, q_i in zip(p, q):

        # caluclate the square of the difference of ith distr elements
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

        # append 
        list_of_squares.append(s)

    # calculate sum of squares
    sosq = sum(list_of_squares)    

    return math.sqrt(sosq)/math.sqrt(2)

## functions added for isca rebuttal--> to be sure keeping things separated and re-writing the function
## mainly related to the computation of Correlation and AR Gap

def compute_hdist(dist_a,dist_b):
	'''
	Function to compute the Hellinger distance between two dictionaries
	Input  : dict_a  -> input dictionary whose Hellinger Distance must be computed
	         dict_b  -> reference or ideal dictionary against which Hellinger distance must be computed
	Output : Hellinger Distance and Correlation (we had used correlation in our Micro Multiprogramming paper)
	''' 
	_in1 = RF.normalize_dict(dist_a.copy())
	_in2 = RF.normalize_dict(dist_b.copy())

	epsilon = 0.00000001
	# update the dictionaries

	for key in _in1.keys():
		if key not in _in2:
			_in2[key] = epsilon # add new entry
	
	for key in _in2.keys():
		if key not in _in1:
			_in1[key] = epsilon # add new entry
	
	# both dictionaries should have the same keys by now
	if set(_in1.keys()) != set(_in2.keys()):
		print('Error : dictionaries need to be re-adjusted')

	## normalize the dictionaries

	_in1 = RF.normalize_dict(_in1)
	_in2 = RF.normalize_dict(_in2)

	list_of_squares = []
	for key,p in _in1.items():
		for _key,q in _in2.items():
			if key == _key:
				s = (math.sqrt(p) - math.sqrt(q)) ** 2
				list_of_squares.append(s)
				break
	# calculate the sum of squares
	sosq = sum(list_of_squares)
	hdist = math.sqrt(sosq)/math.sqrt(2)
	corr = 1-hdist	
	return hdist,corr
				
## functions to evaluate the expectation value
def compute_weight_matrix(_G):
    '''
    Function to compute the weight matrix of a input graph (in the graph format) used for max cut problems
    Input : Graph object
    Output: nxn weight matrix where n is the number of nodes in the graph
    '''
    n = len(_G.nodes())
    w = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            temp = _G.get_edge_data(i,j,default=0)
            if temp != 0:
                w[i,j] = temp['weight']
    return w

def compute_cost_of_cut(graph_cut,weight_matrix):
    '''
    Function to compute the cost of a cut
    Input  : graph_cut (the graph cut in the form of 0 and 1), weight matrix (computed from the graph connections using above function)
    Output : cut value for the cut
    '''
    
    n=len(graph_cut)
    cost = 0
    for i in range(n):
        for j in range(n):
            
            cost = cost + weight_matrix[i,j]* int(graph_cut[i])* (1- int(graph_cut[j]))
    
    return cost


def compute_expected_value(_out_dict,in_graph):
    '''
    Function to compute the expected value of a distribution
    Input : _out_dict  ->  output distribution 
            in_graph   ->  graph used for the max cut problem 
    Output: expected_value
    ''' 
    # check if cut is valid
    
    out_dict = RF.normalize_dict(_out_dict.copy())
    W = compute_weight_matrix(in_graph)
    E = 0
    for key in out_dict:
        key_lst=[] 
        key_lst[:0]=key 
        cost = compute_cost_of_cut(key_lst,W)
        E += out_dict[key]*cost
    
    return E

def obtain_approximation_ratio(_out_dict,in_graph,solution):
	'''
	Function to compute the approximation ratio of a Max Cut problem
	Input  : _out_dict -> output dictionary from the execution of the max cut problem 
	         in_graph  -> input graph for the problem 
	         solution  -> best solution for MaxCut
	Output : Approximation Ratio (AR)
	'''
	## obtain value of cost function from mean of all samples
	#print(solution)
	W = compute_weight_matrix(in_graph)
	mean_from_all_samples = compute_expected_value(_out_dict,in_graph)
	best_cut_value = compute_cost_of_cut(solution,W)
	#print(mean_from_all_samples,best_cut_value)
	
	return mean_from_all_samples/best_cut_value

def norm(numbers): 
    '''
    Function to compute the norm of a list of numbers
    '''

    if isinstance(numbers,list)==1:
        sum_of_numbers = 0 
        for i in numbers:
            sum_of_numbers = sum_of_numbers + math.pow(i,2)
        return math.sqrt(sum_of_numbers)
    else:
        return math.sqrt(math.pow(numbers,2))

def tvd_two_dist(p,q):
    '''
    Function to compute the Total Variation Distance (TVD) between two prob. density functions 
    Input  : two pdfs p and q
    Output : TVD
    '''
    _p = p.copy()
    _p = RF.normalize_dict(_p)
    _q = q.copy()
    _q = RF.normalize_dict(_q)
    
    epsilon = 0.0000001
    ## match both dictionaries
    for key in _p.keys():
        if key not in _q.keys():
            _q[key] = epsilon
    
    for key in _q.keys():
        if key not in _p.keys():
            _p[key] = epsilon

    _p = RF.normalize_dict(_p)
    _q = RF.normalize_dict(_q)

    _q_rearranged = {}
    for key,value in _p.items():
        _q_rearranged[key] = _q[key]

    ## compute_tvd
    tvd = 0 
    for key,value in _p.items():
        diff = value - _q_rearranged[key]
        tvd = tvd + norm(diff)
    return tvd/2
