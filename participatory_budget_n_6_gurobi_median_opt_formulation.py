import cvxpy as cp
import numpy as np
import math,random

import gurobipy as gp
from gurobipy import GRB

from permutations import *

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


n = 6 ##no. of voters used for PD




mapping_3_subset_list = findsubsets(range(n),3) ###denotes the set of all 3 sized subsets of [n]
mapping_all_subset_list = list(powerset(range(n))) ###denotes the set of all subsets of [n]
mapping_3_subset_dict = {} ###denotes a dict of all 3 sized subsets with each set mapping to an index in 20
mapping_all_subset_dict = {}###denotes a dict of all 3 sized subsets with each set mapping to an index in 64

thresh = 1.80 ##finding the optimal val



count = 0
for i in mapping_3_subset_list:
    mapping_3_subset_dict[i]= count
    count += 1

count = 0
for i in mapping_all_subset_list:
    mapping_all_subset_dict[frozenset(i)]=count
    count += 1

    


current_max = 0
current_idx = 0

###Note that $P$ denotes $[6]$ here.

###random_list = np.random.choice(2**(ncr(n,3)), 1000)

##print ("random list",random_list)

non_perm_list = list(unique_permutation()) ##denotes the list of unique cases removed after permuting voter preferences obtained from output of file permutations.py

##non_perm_list = [8190]
print ("Cases to solve obtained",len(non_perm_list))



for index in range(len(non_perm_list))[::-1]:

    itr = non_perm_list[index] ##denotes the specific sub-problem being solved
    model = gp.Model('RAP')
    model.setParam('NonConvex', 2)
    model.setParam('MIPGapAbs',5e-7)
    ##model.setParam('MIPFocus', 1)
    ##model.setParam('MIPGap',1.2)
   
###First part common for all no use of itr
    X= model.addVars(2**n, vtype=GRB.CONTINUOUS,name= "X_name") ##denotes X(S) for every subset of $[n]$. 


    
    Z = model.addVars(ncr(n,3),2**n, vtype=GRB.CONTINUOUS,name = "Z_name")

    V = model.addVars(2**n, vtype=GRB.CONTINUOUS,name = "V_name") ##denotes V(S) for every subset of $[n]$. 

    
    
    


    for i in range(n):
        
        temp_S = set(range(n))- set({i}) ### set of all elements excluding voter $i$ 
        
        
        temp_list= list(powerset(temp_S)) ###powerset of all elements excluding voter $i$

        testing_list = [] ###stores X(iUS) for all subsets of S in [n]-i

        for j in temp_list:
          testing_list.append(X[mapping_all_subset_dict[frozenset(set(j)|set({i}))]])
        
        
        
        model.addConstr(gp.quicksum(testing_list) == 1,name="individualvote"+str(i)) ###\sum_{S \ni i} X(S) = 1

       

    model.addConstrs((Z.sum(j,'*') == 1 for j in range(ncr(n,3)))) ## for every subset of budgets, \{\{x,y\},c}\, we have $\sum_{S \in \mathcal{P}(P)} Z^{\c,\{x,y\}}(S) = 1$
    model.addConstr(V.sum('*') == 1) ### \sum_{S \in \mathcal{P}([6])} V[S] = 1
    model.addConstrs((X[j]>=V[j] for j in range(2**n))) ## X(S) \geq V(S)
    
    for i in range(ncr(n,3)):
        Q_set = set(mapping_3_subset_list[i]) ## denotes the precise set $Q$ chosen
        Q_set_list = list(Q_set)
        
        Q_set_comp = list(powerset(set(range(n))-Q_set)) ###power set of the set $[n]-Q$
        
       
        model.addConstrs((X[j]>=Z[i,j] for j in range(2**n))) ##denoting X(S) \geq Z^{\{x,y\},c} 
            
        
        
        
        for j in Q_set_comp: ###considers every set $Q$
            
                model.addConstr(Z[i,mapping_all_subset_dict[frozenset(set(j)|Q_set)]]== X[mapping_all_subset_dict[frozenset(set(j)|Q_set)]]) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing all elts of $Q$.
            
            
                model.addConstr(Z[i,mapping_all_subset_dict[frozenset(set(j))]]==0) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing no elements of $R$.
            
            
               
            
            
            
              

                    
            
            
            
            
            
    list_temp1 = [] ###calculation the distance of bargaing solution Z with other voters not ones in Q excluding constant 1


    for Q_index in range(ncr(n,3)):
        for i in set(range(n))-set(mapping_3_subset_list[Q_index]):
            for S in powerset(set(range(n))-set({i})):
                
                    list_temp1.append(-2*Z[Q_index,mapping_all_subset_dict[frozenset(set(S)|set({i}))]])
    list_temp_const_1 = []
    list_temp_const_2 = []

    for Q_index in range(ncr(n,3)): ##calculating the constants 
        for i in set(range(n))-set(mapping_3_subset_list[Q_index]):
            
                list_temp_const_1.append(2)


                

    list_temp2 = [] ###calculation the distance of optimal solution with all voters excluding constant 1
    for i in range(n): ##distance with voter $i$
        for S in powerset(set(range(n))-set({i})): 
            list_temp2.append(-2*V[mapping_all_subset_dict[frozenset(set(S)|set({i}))]])
            
    for i in range(n): ##constants
        list_temp_const_2.append(2)
            
    
    model.setObjective((1.0/(ncr(n,3)*(n-3)))*(gp.quicksum(list_temp1)+gp.quicksum(list_temp_const_1)) - thresh*(1.0/n)*(gp.quicksum(list_temp2)+gp.quicksum(list_temp_const_2)),GRB.MAXIMIZE)




    ####Second part which uses itr begins here












    
    list_diff_Q = format(itr, '0'+str(ncr(n,3))+'b')   ##representation of set Q in binary format

    for j in range(ncr(n,3)): ###denoting which set Q is considered
        list_temp3 = [] ### computing X(xy) + X(xyc) + X(xc)+ X(yc) on a incremental space of x,y,c. 
        for i in range(2,4):
                for Q_prime in findsubsets(set(mapping_3_subset_list[j]),i):
                    for V_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                        list_temp3.append(X[mapping_all_subset_dict[frozenset(set(Q_prime)|set(V_prime))]])
        
        if (list_diff_Q[j]=='1'): ## considers Case 2 as defined in every condition. 
            model.addConstr(gp.quicksum(list_temp3) >=1) ##puts desired constraint on X(xy) + X(xyc) + X(xc)+ X(yc) on a incremental space of x,y,c 
            for Q_prime in findsubsets(set(mapping_3_subset_list[j]),1): ### identifying the disagreement point
                    for U_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                      
                        model.addConstr(Z[j,mapping_all_subset_dict[frozenset(set(Q_prime)|set(U_prime))]]==0) ## Z(x)=Z(y)= Z(c) = 0 in the incremental budget space

                        
                        
            ##print("entered and finished part1",j)
        elif (list_diff_Q[j]=='0'):
            ##print ("entering here",list_diff_R[j])
            model.addConstr(gp.quicksum(list_temp3) <=1)
            for Q_prime in findsubsets(set(mapping_3_subset_list[j]),2):
                    for U_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                      
                        model.addConstr(Z[j,mapping_all_subset_dict[frozenset(set(Q_prime)|set(U_prime))]]==X[mapping_all_subset_dict[frozenset(set(Q_prime)|set(U_prime))]]) ## Z(xy) = X(xy); Z(yc)= X(yc); Z(xc)=X(xc) = 0 in the incremental budget space


                        
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

            file1 = open("myfile_gurobi_median_upto_toggle_updated.txt","a")
            file1.writelines(str(model.status)+ " case "+str(itr)+ " itr " + str(index)+ " not solved \n",)
            file1.close()
            
            ##print (model.status,"iteration",itr,)
    
    if (model.status == 2):
        file1 = open("myfile_gurobi_median_upto_toggle_updated.txt","a")
        file1.writelines(str(model.status)+ " case "+str(itr)+ " itr " + str(index)+ " solved " +" Obj Value " + str (model.objVal)+"\n")
        file1.close()


    if (itr % 1 == 0):
        
            print ("Current max", current_max,"max_idx",current_idx, "iteration",index,"case number",itr)
            
    if (model.ObjVal > current_max):
      current_max = max(model.objVal,current_max) ##checking maximum value obtained
      current_idx = itr

      

    
    if (current_max > 5e-6):
      print ("Not working", current_max)
    
print ("Final maximum obtained",current_max,"max_idx",current_idx)

          


###random_list = [213]








##prob.solve()


