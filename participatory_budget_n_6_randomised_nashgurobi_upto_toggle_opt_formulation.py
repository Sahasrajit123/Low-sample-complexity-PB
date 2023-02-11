import cvxpy as cp
import numpy as np
import math,random

import gurobipy as gp
from gurobipy import GRB

import itertools
from itertools import combinations, chain
 
def findsubsets(s, n):
    return list((map(frozenset, itertools.combinations(s, n))))


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

import asyncio

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


n = 6




mapping_3_subset_list = findsubsets(range(n),3)
mapping_all_subset_list = list(powerset(range(n)))
mapping_3_subset_dict = {}###use this for mapping set to a corresponding index 
mapping_all_subset_dict = {}###use this for mapping set to a corresponding index 

thresh = 1.66 ##finding the optimal val



count = 0
for i in mapping_3_subset_list:
    mapping_3_subset_dict[i]= count
    count += 1

count = 0
for i in mapping_all_subset_list:
    mapping_all_subset_dict[frozenset(i)]=count
    count += 1

    
##mapping_subset_dict[frozenset({0,1,2})]=0    
##print (mapping_3_subset_dict)
##print(mapping_all_subset_dict)




##X_curr_soln = np.zeros(2**n)
##Z_curr_soln = np.zeros((ncr(n,3)*3,2**n))
##Z_v_curr_soln = np.zeros(2**n)

current_max = -1e-5
current_idx = 0


non_perm_upto_toggle = [262135, 16311, 262141, 16319, 8191, 16335, 16351, 16375, 97789, 16379, 524287, 97791, 16382, 16383, 228094, 32743, 32752, 32753, 32755, 32756, 32757, 32759, 1048575, 120687, 32764, 32765, 32767, 1023, 120695, 31220, 89839, 89845, 89855, 1535, 120811, 120812, 32511, 120813, 120815, 120819, 120821, 120823, 120829, 98035, 81755, 98038, 98039, 98043, 98046, 1791, 81790, 65520, 65521, 65523, 65527, 2031, 65535, 2047, 81847, 98133, 81851, 98135, 81854, 98137, 81855, 98147, 81867, 81870, 81871, 98163, 98165, 81883, 98166, 81885, 81886, 98169, 98173, 98175, 90038, 90039, 90043, 90045, 90046, 90047, 90052, 81918, 81919, 90061, 90063, 90070, 90071, 90077, 90078, 31475, 90093, 90094, 31479, 90095, 31483, 90110, 90111, 98275, 98277, 98279, 98281, 98287, 98291, 98293, 98295, 98296, 98297, 98299, 98300, 98301, 98303, 122742, 76983, 76987, 76991, 11455, 77007, 77019, 77021, 77022, 77023, 11487, 11511, 11515, 11518, 3327, 77055, 11519, 77054, 77135, 77147, 77149, 77151, 11615, 77165, 77166, 77167, 11631, 11639, 11643, 11645, 11646, 77183, 11647, 77182, 3455, 3519, 3551, 11751, 11755, 77292, 11757, 77293, 77295, 3567, 11759, 11763, 11765, 11767, 11769, 11771, 11772, 11773, 3582, 11775, 77311, 3583, 122835, 122836, 122837, 122839, 122841, 122843, 122844, 122845, 122847, 122866, 122867, 122870, 122871, 122873, 122874, 122875, 122878, 122879, 3903, 77643, 77647, 12119, 77659, 241500, 12123, 77660, 3935, 77661, 12125, 12127, 241501, 241503, 77663, 241514, 241515, 241518, 241519, 12147, 12150, 12151, 12153, 12154, 12155, 12158, 12159, 241535, 3966, 3967, 77694, 241534, 77695, 31728, 3999, 31731, 31733, 131056, 131057, 4023, 4027, 4030, 4031, 131059, 4047, 131063, 131064, 4059, 520157, 4061, 520159, 4062, 4063, 131065, 241643, 241647, 12273, 12275, 12279, 12280, 12281, 12283, 131071, 4094, 4095, 12287, 77823, 241663, 520191, 78047, 78078, 78079, 12543, 88794, 88797, 88798, 78143, 78175, 78191, 78206, 78207, 12671, 88823, 78255, 78263, 78267, 78269, 78270, 78271, 88829, 88830, 78287, 88831, 78301, 78302, 78303, 78317, 78318, 78319, 12783, 12791, 12797, 78334, 78335, 12799, 97011, 97013, 97015, 243567, 78623, 242467, 242471, 242479, 242487, 78647, 78651, 78654, 242495, 78655, 13119, 119623, 119631, 78671, 119639, 78683, 78685, 78686, 119647, 78687, 242543, 119671, 13175, 13179, 78718, 119679, 242559, 78719, 13183, 78735, 78743, 78747, 78749, 78750, 78751, 78774, 78775, 78777, 78778, 78779, 78782, 78783, 78795, 78798, 78799, 78811, 78812, 78813, 78814, 78815, 13299, 13302, 119799, 13303, 78846, 119807, 242687, 78847, 13311, 79069, 79070, 79071, 13535, 79086, 79087, 13551, 13559, 13563, 13565, 79102, 13566, 79103, 13567, 79159, 79163, 79165, 79167, 13631, 243687, 79197, 79198, 79199, 243693, 243695, 79213, 79214, 13679, 79215, 13687, 13691, 13693, 79230, 13694, 79231, 13695, 243699, 79271, 79277, 79278, 79279, 79287, 79292, 79293, 79294, 79295, 79309, 79310, 79311, 79324, 79325, 79326, 79327, 13799, 79340, 13805, 79341, 13807, 79343, 79342, 13811, 13813, 13814, 13815, 13820, 13821, 79358, 79359, 13823, 79451, 79454, 79455, 13919, 13943, 13947, 13949, 79486, 79487, 13951, 13950, 15801, 7661, 7662, 79511, 15804, 7663, 79515, 79517, 79519, 79542, 79543, 14007, 79545, 79546, 14011, 79548, 14013, 14014, 79551, 14015, 79549, 79547, 79550, 79563, 79565, 79566, 14031, 79567, 14039, 79579, 79580, 14045, 79582, 79583, 14047, 79581, 7678, 14055, 14059, 79596, 79597, 14061, 79598, 14062, 14063, 7679, 14067, 79599, 14069, 14070, 14071, 14073, 14074, 14075, 14076, 14077, 79614, 14078, 79615, 14079, 79639, 79643, 79645, 79646, 79647, 79655, 79659, 79661, 79662, 79663, 14127, 79670, 14135, 243511, 79673, 79674, 79675, 79671, 14141, 79678, 79677, 15837, 243519, 79679, 14143, 15836, 79676, 79691, 79693, 79694, 79695, 120663, 79707, 79708, 79709, 79710, 79711, 120671, 243558, 14183, 120678, 120679, 243559, 14187, 79724, 14189, 14190, 14191, 79725, 79726, 243566, 120686, 14195, 79727, 120694, 14197, 14198, 14199, 14201, 14202, 14203, 14204, 14205, 243583, 120703, 79742, 14206, 14207, 243579, 120702, 243582, 79743, 79755, 79757, 79759, 15853, 79765, 15854, 79767, 79772, 79773, 79774, 79775, 97268, 79781, 79782, 79783, 79785, 79786, 79787, 79788, 79789, 79790, 79791, 15859, 15860, 79798, 79799, 79800, 79801, 79802, 79803, 79804, 79805, 79806, 15861, 15862, 79807, 15864, 79817, 79819, 79820, 79821, 79822, 79823, 15865, 15866, 15867, 79835, 79836, 79837, 79838, 79839, 14307, 243683, 14309, 14310, 14311, 120803, 120805, 120807, 15870, 79852, 14316, 79853, 14317, 79854, 79855, 14319, 14321, 14323, 14324, 14325, 14326, 14327, 243702, 243703, 120820, 14332, 14333, 79870, 14335, 79871, 243708, 243709, 243711, 120827, 120828, 120831, 32245, 88519, 88525, 88527, 88534, 88535, 88540, 88541, 88542, 88543, 31203, 96741, 31205, 96743, 31207, 96745, 96747, 96749, 88557, 227823, 96751, 31215, 88559, 31219, 88558, 96757, 31221, 96759, 31223, 96761, 31217, 96763, 31228, 96765, 31229, 88575, 227839, 96767, 31231, 88574, 96764, 88663, 88670, 88671, 88702, 88703, 228023, 228025, 228027, 228029, 228031, 88775, 88779, 88782, 228047, 88783, 96979, 31443, 88789, 88790, 31446, 96983, 88793, 96985, 96986, 88796, 31447, 88791, 96982, 96987, 88795, 228061, 96990, 228063, 96991, 31455, 88799, 88807, 88806, 88810, 88811, 88814, 228079, 88815, 31473, 31474, 97010, 88820, 88821, 88822, 31477, 97014, 88825, 88826, 88827, 88828, 31478, 97016, 97017, 97018, 31482, 97020, 97019, 97021, 97022, 31487, 228095, 97023, 97091, 97095, 97099, 97103, 97107, 97108, 97109, 97111, 97113, 97115, 97116, 97117, 97119, 97122, 97123, 97126, 97127, 97130, 97131, 97133, 97134, 97135, 97139, 97140, 97141, 97142, 97143, 97145, 97146, 97147, 97148, 97149, 97150, 97151, 88979, 88981, 88983, 88985, 88986, 88987, 88988, 88989, 88990, 88991, 88999, 89001, 89003, 89005, 89007, 89012, 89013, 89014, 89015, 89018, 89019, 89020, 89021, 89022, 89023, 89029, 89030, 89031, 89037, 89038, 89039, 89046, 89047, 89052, 89053, 89054, 89055, 97249, 261091, 97251, 97253, 97255, 261095, 97257, 97259, 97261, 89069, 89071, 97263, 228335, 261103, 97267, 31729, 97269, 89070, 31735, 97272, 261111, 97273, 97271, 229375, 97276, 89086, 89087, 31743, 97275, 261117, 97277, 261119, 228351, 97279, 7295, 245687, 245691, 245693, 245695, 7359, 7391, 15606, 15607, 15609, 15610, 15611, 7422, 81150, 81151, 15615, 7423, 15614, 245724, 245725, 245727, 81260, 81261, 81262, 81263, 15733, 15734, 15735, 15740, 15741, 81278, 81279, 15743, 15742, 245739, 7583, 245742, 245743, 7599, 15795, 15797, 7607, 81335, 15799, 81337, 7611, 81339, 7613, 7614, 81341, 81343, 15807, 7615, 15803, 15805, 81340, 81353, 81355, 81356, 81357, 81358, 81359, 15827, 15829, 15830, 89559, 15831, 81371, 89564, 81372, 81373, 81374, 89567, 81375, 15839, 15843, 89565, 15845, 15846, 15847, 15849, 15850, 15851, 89580, 89581, 89582, 15852, 81388, 81389, 89583, 81391, 32244, 15855, 81390, 32243, 245758, 97783, 32247, 97787, 32252, 15868, 15869, 15863, 245759, 97788, 32253, 89598, 89599, 15871, 32255, 81406, 228863, 16183, 89726, 89727, 81336, 8127, 229047, 229048, 229049, 32499, 229051, 229052, 229053, 229055, 32503, 229069, 229071, 89812, 89813, 89814, 89815, 16223, 89818, 89819, 89820, 229084, 229085, 229087, 89823, 89821, 89822, 89829, 89830, 89831, 89834, 89835, 89836, 89837, 89838, 229101, 32496, 32497, 32498, 229102, 89844, 229103, 89846, 89847, 32504, 32502, 89850, 98040, 89852, 89853, 89854, 32505, 32506, 98041, 98042, 89851, 32507, 32510, 229118, 229119, 98047, 81683, 81685, 81686, 81687, 81689, 81690, 81691, 81692, 81693, 81694, 81695, 16179, 16182, 81718, 81720, 81721, 16185, 81722, 16186, 16187, 16190, 81719, 81723, 81726, 81727, 16191, 98115, 98117, 98119, 81737, 81738, 98123, 81739, 98125, 81742, 98127, 81743, 16211, 98132, 16213, 16214, 16215, 98131, 16217, 16218, 81754, 98140, 16219, 81756, 16220, 81757, 16221, 81758, 16222, 98139, 98141, 98150, 98143, 81759, 98153, 98154, 98151, 98156, 98149, 98155, 98157, 98158, 16241, 16242, 16243, 98164, 98159, 16246, 98167, 122743, 16248, 16249, 16250, 16247, 16251, 245630, 245631, 122751, 16254, 81791, 98170, 98171, 98172, 122750, 98174, 16255, 8118, 8079, 8119, 81811, 81812, 81813, 81814, 8087, 81816, 81815, 81817, 81819, 81820, 8093, 8094, 81818, 81821, 81822, 81823, 8095, 90021, 8123, 90023, 90024, 90025, 90026, 90027, 90029, 90031, 16305, 16306, 16307, 90036, 90037, 81846, 16310, 81848, 16312, 81849, 16313, 81850, 16314, 16315, 16318, 90042, 90044, 8122, 16323, 8126, 90053, 16326, 90054, 81864, 81865, 16329, 81866, 16331, 90060, 90062, 90055, 16327, 16337, 16338, 16339, 16340, 16341, 16342, 16343, 16344, 16345, 81882, 16346, 16347, 81884, 16348, 16349, 32736, 16350, 81887, 90076, 90079, 32737, 32739, 32741, 98280, 81407, 245738, 98283, 90092, 98284, 98285, 229357, 16368, 16369, 16370, 16371, 32751, 229359, 524286, 16374, 16376, 16377, 16378, 131067, 262140, 98292, 8190, 262143]

## cases obtained after removing same cases obtained after permutation of voter prefernces and toggling the all the bits 
##correspoding to a case for example 2^20-1 and 0 are obtained after toggling all bits correponding to a case hence one may be solved.

###these cases are output from file toggle_permutations.py




for index in range(len(non_perm_upto_toggle))[::-1]:


    itr = non_perm_upto_toggle[index]
    model = gp.Model('RAP')
    model.setParam('NonConvex', 2)
    model.setParam('MIPGapAbs',5e-7)
    ##model.setParam('MIPFocus', 1)
    ##model.setParam('MIPGap',1.2)
   
###First part common for all no use of itr
    X= model.addVars(2**n, vtype=GRB.CONTINUOUS,name= "X_name") ##denotes X(S) for every subset of $[n]$. 
###Z = cp.Variable((ncr(n,3)*3,2**n))

    ## no of terms 'ncr(n,3)*3' ## suppose the index is $i$, with $i/3$ denoting the subset number $Q$ in code and $i mod 3$ denoting the dis-agreement 
    ##point index in $Q$ like if $i%3 =0$, the first element of Q is diagreement point and so on..

    Z = model.addVars(ncr(n,3)*3,2**n, vtype=GRB.CONTINUOUS,name = "Z_name")
##Z_v = cp.Variable(2**n)
    V = model.addVars(2**n, vtype=GRB.CONTINUOUS,name = "V_name") ##denotes V(S) for every subset of $[n]$. 

    ## no of terms 'ncr(n,3)*3' ## suppose the index is $i$, with $i/3$ denoting the subset number $Q$ in code and $i mod 3$ denoting the dis-agreement 
    ##point index in $Q$ like if $i%3 =0$, the first element of Q is diagreement point and so on..

    alpha_1 = model.addVars(ncr(n,3)*3, lb=0.0, ub= 1.0, vtype=GRB.CONTINUOUS,name= "alpha1_name") ##denotes $\alpha_{\{\{x,y\},c}\}$
    alpha_2 = model.addVars(ncr(n,3)*3, lb=0.0, ub= 1.0, vtype=GRB.CONTINUOUS,name = "alpha2_name") ##denotes $\beta_{\{\{x,y\},c}\}$



    
    


    for i in range(n):
        temp_S = set(range(n))- set({i}) ### set of all elements excluding voter $i$ 
        
        
        temp_list= list(powerset(temp_S)) ###powerset of all elements excluding voter $i$
        
        testing_list = [] ###stores X(iUS) for all subsets of S in [n]-i

        for j in temp_list:
          testing_list.append(X[mapping_all_subset_dict[frozenset(set(j)|set({i}))]])
        
        
        
        model.addConstr(gp.quicksum(testing_list) == 1,name="individualvote"+str(i)) ###\sum_{S \ni i} X(S) = 1

       

    model.addConstrs((Z.sum(j,'*') == 1 for j in range(ncr(n,3)*3))) ## for every subset of budgets, \{\{x,y\},c}\, we have $\sum_{S \in \mathcal{P}(P)} Z^{\c,\{x,y\}}(S) = 1$
    model.addConstr(V.sum('*') == 1) ### \sum_{S \in \mathcal{P}([6])} V[S] = 1
    model.addConstrs((X[j]>=V[j] for j in range(2**n))) ## X(S) \geq V(S)
    
    for i in range(ncr(n,3)):
        R_set = set(mapping_3_subset_list[i]) ## denotes the precise set $Q$ chosen
        R_set_list = list(R_set)
        
        Q_set_excl = list(powerset(set(range(n))-R_set)) ###power set of the set $[n]-Q$
        
        for k in range(3): ###denotes the position of the disagreement point in the set $Q$

            model.addConstrs((X[j]>=Z[i*3+k,j] for j in range(2**n))) ##denoting X(S) \geq Z^{\{x,y\},c} 
            
        
            barg_set = set(R_set-{R_set_list[k]}) ## set of bargaining voters
        
            
        
        ##print (temp_list)
            for j in Q_set_excl: ###considers every set $Q$
            
                model.addConstr(Z[3*i+k,mapping_all_subset_dict[frozenset(set(j)|R_set)]]== X[mapping_all_subset_dict[frozenset(set(j)|R_set)]]) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing all elts of $Q$.
            
            
                model.addConstr(Z[3*i+k,mapping_all_subset_dict[frozenset(set(j))]]==0) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing no elements of $Q$.
            
            
            for j in Q_set_excl: ###corresponds to Z(xy) =X(xy) on a incremental space of x,y,c.
            
                model.addConstr(Z[3*i+k,mapping_all_subset_dict[frozenset(set(j)|barg_set)]]== X[mapping_all_subset_dict[frozenset(set(j)|barg_set)]]) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing the bargaining elements of $Q$ i.e. x & y but not $c$.
            
            for j in Q_set_excl: ##corresponds to Z(c) = 0
                model.addConstr(Z[3*i+k,mapping_all_subset_dict[frozenset(set(j)|{R_set_list[k]})]]==0) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing the disagreement point of $Q$ i.e. c but not $x,y$.
                
            
            
            list_temp3 = [] ###getting X(ab)+X(bc) + X(ca)+X(abc) on the incremental budget space of x,y and c.
            for m in range(2,4): ###getting X(ab)+X(bc) + X(ca)+X(abc)
                    for R_prime in findsubsets(set(mapping_3_subset_list[i]),m):
                        for V_prime in Q_set_excl:
                            list_temp3.append(X[mapping_all_subset_dict[frozenset(set(R_prime)|set(V_prime))]])
                            
            
            
            
            
          
            
            for j in barg_set: ###for each of Z(x) and Z(y) note that since the cases treat $x$ and $y$ equivalently, we just denote it by $x$ in the loop.
                
                    ###print ("j",j)
                    ##list_temp4 = []
                    list_temp_z_x = [] ## capturing $Z^{{\x,y\},c}(x)$ on the incremental space of x,y and $c$.
                    list_temp_z_xc = [] # capturing $Z^{{\x,y\},c}(xc)$ on the incremental space of x,y and $c$.

                    list_temp_x_xc = [] # capturing $X(xc)$ on the incremental space of x,y and $c$.

                    
                    
                    for l in Q_set_excl: ##considering all posibilities for expansion of Z(a)
                        
                        ##print ("frozen sets",frozenset(set(l)|set({2})))
                
                        list_temp_z_x.append(Z[3*i+k,mapping_all_subset_dict[frozenset(set({j})|set(l))]]) ## capturing $Z^{{\x,y\},c}(x)$ on the incremental space of x,y and $c$.
                            ##i.e. $Z^{{\x,y\},c}(S)$ for every S containing x but not y, not c.  

                        list_temp_z_xc.append(Z[3*i+k,mapping_all_subset_dict[frozenset(set({j})|set(l)|set({R_set_list[k]}))]]) ## capturing $Z^{{\x,y\},c}(xc)$ on the incremental space of x,y and $c$.
                            ##i.e. $Z^{{\x,y\},c}(S)$ sum over every S containing x,c but not y.  

                        list_temp_x_xc.append(X[mapping_all_subset_dict[frozenset(set({j})|set(l)|set({R_set_list[k]}))]]) ## capturing $X(xc)$ on the incremental space of x,y and $c$.
                            ##i.e. $X(S)$ sum over S containing x,c but not y.  
                        
                        

                    
                    model.addConstr(gp.quicksum(list_temp_z_x+list_temp_z_xc)==0.5*(1-gp.quicksum(list_temp3))+gp.quicksum(list_temp_x_xc)) ###enforcing Z(x)+Z(xc)=k_a ##incremental budget space of x,y,c

              

                    
            
            
            
            
            
            

    list_temp1 = [] ###calculation the distance of bargaing solution Z with other voters not ones in Q excluding constant 1


    for R_index in range(ncr(n,3)):
        for i in set(range(n))-set(mapping_3_subset_list[R_index]):
            for S in powerset(set(range(n))-set({i})):
                for j in range(3): ##all possible orderings of Nash(a,b,c)- only position of c matters 
                    list_temp1.append(-2*Z[3*R_index+j,mapping_all_subset_dict[frozenset(set(S)|set({i}))]])
    list_temp_const_1 = []
    list_temp_const_2 = []

    for R_index in range(ncr(n,3)): ##calculating the constants 
        for i in set(range(n))-set(mapping_3_subset_list[R_index]):
            for j in range(3):
            
                list_temp_const_1.append(2)


                

    list_temp2 = [] ###calculation the distance of optimal solution with all voters excluding constant 1
    for i in range(n): ##distance with voter $i$
        for S in powerset(set(range(n))-set({i})): 
            list_temp2.append(-2*V[mapping_all_subset_dict[frozenset(set(S)|set({i}))]])
            
    for i in range(n): ##constants
        list_temp_const_2.append(2)
            
    
    model.setObjective((1.0/(ncr(n,3)*3*(n-3)))*(gp.quicksum(list_temp1)+gp.quicksum(list_temp_const_1)) - thresh*(1.0/n)*(gp.quicksum(list_temp2)+gp.quicksum(list_temp_const_2)),GRB.MAXIMIZE)




    ####Second part which uses itr begins here












    
    list_diff_R = format(itr, '0'+str(ncr(n,3))+'b') ###analysing every case in $\mathbb{K}$ after removing the directly toggled cases and those unique to permutation.
    
    ##representation in binary format

    for j in range(ncr(n,3)): ###denoting which set R is considered
        list_temp3 = [] ### computing X(xy) + X(xyc) + X(xc)+ X(yc) on a incremental space of x,y,c. 
        for i in range(2,4):
                for R_prime in findsubsets(set(mapping_3_subset_list[j]),i):
                    for V_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                        list_temp3.append(X[mapping_all_subset_dict[frozenset(set(R_prime)|set(V_prime))]])
        
        if (list_diff_R[j]=='1'): ## considers Case 2 as defined in every condition. 
            model.addConstr(gp.quicksum(list_temp3) >=1) ##puts desired constraint on X(xy) + X(xyc) + X(xc)+ X(yc) on a incremental space of x,y,c 
            for R_prime in findsubsets(set(mapping_3_subset_list[j]),1): ### identifying the disagreement point
                    for U_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                      for l in range(3):
                        model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset(set(R_prime)|set(U_prime))]]==0) ## Z(x)=Z(y)= Z(c) = 0 in the incremental budget space

                        if (list(mapping_3_subset_list[j])[(l+1)%3] in R_prime): ##choosing one disagreement point x
                          model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(R_prime))|set(U_prime))]]==alpha_1[3*j+l]*X[mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(R_prime))|set(U_prime))]]) ###Z(yc) = alpha*X(yc) in incremental budget space of $\{x,y,c\}$

                        elif (list(mapping_3_subset_list[j])[(l+2)%3] in R_prime): ##choosing other disagreement point y
                          model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(R_prime))|set(U_prime))]]== alpha_2[3*j+l]*X[mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(R_prime))|set(U_prime))]]) ###Z(xc) = beta*X(xc) in incremental budget space of $\{x,y,c\}$
                        
                        
            ##print("entered and finished part1",j)
        elif (list_diff_R[j]=='0'):
            ##print ("entering here",list_diff_R[j])
            model.addConstr(gp.quicksum(list_temp3) <=1)
            for R_prime in findsubsets(set(mapping_3_subset_list[j]),2):
                    for U_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                      for l in range(3):
                        model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset(set(R_prime)|set(U_prime))]]==X[mapping_all_subset_dict[frozenset(set(R_prime)|set(U_prime))]]) ## Z(xy) = X(xy); Z(yc)= X(yc); Z(xc)=X(xc) = 0 in the incremental budget space


                        if (list(mapping_3_subset_list[j])[(l+1)%3] in R_prime and list(mapping_3_subset_list[j])[l] in R_prime): ##R_prime is one pair of points ${x,c}$
                            model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(R_prime))|set(U_prime))]]== alpha_1[3*j+l]*X[mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(R_prime))|set(U_prime))]]) ###Z(y) = alpha*X(y)

                        elif (list(mapping_3_subset_list[j])[(l+2)%3] in R_prime and list(mapping_3_subset_list[j])[l] in R_prime): ##R_prime is one pair of points ${y,c}$
                            model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(R_prime))|set(U_prime))]]== alpha_2[3*j+l]*X[mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(R_prime))|set(U_prime))]]) ###Z(x) = beta*X(x)

            ##print("entered and finished part2",j)
        else:
            print ("The list contains spaces neither 0 nor 1")
            break
            break
 
    
    model.write("RAP.lp")
    print ("itr")
    model.optimize()
    
    if (model.status != 2):
            print (model.status,"iteration",itr,"not solved")

            file1 = open("myfile_gurobi_upto_toggle_updated.txt","a")
            file1.writelines(str(model.status)+ " case "+str(itr)+ " itr " + str(index)+ " not solved \n",)
            file1.close()
            
            ##print (model.status,"iteration",itr,)
    
    if (model.status == 2):
        file1 = open("myfile_gurobi_upto_toggle_updated.txt","a")
        file1.writelines(str(model.status)+ " case "+str(itr)+ " itr " + str(index)+ " solved " +" Obj Value " + str (model.objVal)+"\n")
        file1.close()


    if (itr % 1 == 0):
        
            print ("Current max", current_max,"max_idx",current_idx, "iteration",index,"case number",itr)
            
    if (model.objVal > current_max):
      current_max = max(model.objVal,current_max) ##checking maximum value obtained
      current_idx = itr

     

    
    if (current_max > 5e-5):
      print ("Not working case no",itr,"has optimum value", current_max)
    
print ("Final maximum obtained",current_max,"max_idx",current_idx)

          


