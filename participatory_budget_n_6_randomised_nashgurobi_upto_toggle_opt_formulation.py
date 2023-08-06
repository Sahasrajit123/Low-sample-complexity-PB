import cvxpy as cp
import numpy as np
import math,random

import gurobipy as gp
from gurobipy import GRB

import itertools
from itertools import combinations, chain

from toggle_permutations import *
 
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


non_perm_upto_toggle = list(unique_permutations_toggle()) 

## cases obtained after removing same cases obtained after permutation of voter prefernces and toggling the all the bits 
##correspoding to a case for example 2^20-1 and 0 are obtained after toggling all bits correponding to a case hence one may be solved.

###these cases are output from file toggle_permutations.py

print ("Cases to solve obtained",len(non_perm_upto_toggle))





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
        Q_set = set(mapping_3_subset_list[i]) ## denotes the precise set $Q$ chosen
        Q_set_list = list(Q_set)
        
        Q_set_comp = list(powerset(set(range(n))-Q_set)) ###power set of the set $[n]-Q$
        
        for k in range(3): ###denotes the position of the disagreement point in the set $Q$

            model.addConstrs((X[j]>=Z[i*3+k,j] for j in range(2**n))) ##denoting X(S) \geq Z^{\{x,y\},c} 
            
        
            barg_set = set(Q_set-{Q_set_list[k]}) ## set of bargaining voters
        
            
        
        ##print (temp_list)
            for j in Q_set_comp: ###considers every set $Q$
            
                model.addConstr(Z[3*i+k,mapping_all_subset_dict[frozenset(set(j)|Q_set)]]== X[mapping_all_subset_dict[frozenset(set(j)|Q_set)]]) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing all elts of $Q$.
            
            
                model.addConstr(Z[3*i+k,mapping_all_subset_dict[frozenset(set(j))]]==0) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing no elements of $Q$.
            
            
            for j in Q_set_comp: ###corresponds to Z(xy) =X(xy) on a incremental space of x,y,c.
            
                model.addConstr(Z[3*i+k,mapping_all_subset_dict[frozenset(set(j)|barg_set)]]== X[mapping_all_subset_dict[frozenset(set(j)|barg_set)]]) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing the bargaining elements of $Q$ i.e. x & y but not $c$.
            
            for j in Q_set_comp: ##corresponds to Z(c) = 0
                model.addConstr(Z[3*i+k,mapping_all_subset_dict[frozenset(set(j)|{Q_set_list[k]})]]==0) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing the disagreement point of $Q$ i.e. c but not $x,y$.
                
            
            
            list_temp3 = [] ###getting X(ab)+X(bc) + X(ca)+X(abc) on the incremental budget space of x,y and c.
            for m in range(2,4): ###getting X(ab)+X(bc) + X(ca)+X(abc)
                    for Q_prime in findsubsets(set(mapping_3_subset_list[i]),m):
                        for V_prime in Q_set_comp:
                            list_temp3.append(X[mapping_all_subset_dict[frozenset(set(Q_prime)|set(V_prime))]])
                            
            
            
            
            
          
            
            for j in barg_set: ###for each of Z(x) and Z(y) note that since the cases treat $x$ and $y$ equivalently, we just denote it by $x$ in the loop.
                
                    ###print ("j",j)
                    ##list_temp4 = []
                    list_temp_z_x = [] ## capturing $Z^{{\x,y\},c}(x)$ on the incremental space of x,y and $c$.
                    list_temp_z_xc = [] # capturing $Z^{{\x,y\},c}(xc)$ on the incremental space of x,y and $c$.

                    list_temp_x_xc = [] # capturing $X(xc)$ on the incremental space of x,y and $c$.

                    
                    
                    for l in Q_set_comp: ##considering all posibilities for expansion of Z(a)
                        
                        ##print ("frozen sets",frozenset(set(l)|set({2})))
                
                        list_temp_z_x.append(Z[3*i+k,mapping_all_subset_dict[frozenset(set({j})|set(l))]]) ## capturing $Z^{{\x,y\},c}(x)$ on the incremental space of x,y and $c$.
                            ##i.e. $Z^{{\x,y\},c}(S)$ for every S containing x but not y, not c.  

                        list_temp_z_xc.append(Z[3*i+k,mapping_all_subset_dict[frozenset(set({j})|set(l)|set({Q_set_list[k]}))]]) ## capturing $Z^{{\x,y\},c}(xc)$ on the incremental space of x,y and $c$.
                            ##i.e. $Z^{{\x,y\},c}(S)$ sum over every S containing x,c but not y.  

                        list_temp_x_xc.append(X[mapping_all_subset_dict[frozenset(set({j})|set(l)|set({Q_set_list[k]}))]]) ## capturing $X(xc)$ on the incremental space of x,y and $c$.
                            ##i.e. $X(S)$ sum over S containing x,c but not y.  
                        
                        

                    
                    model.addConstr(gp.quicksum(list_temp_z_x+list_temp_z_xc)==0.5*(1-gp.quicksum(list_temp3))+gp.quicksum(list_temp_x_xc)) ###enforcing Z(x)+Z(xc)=k_a ##incremental budget space of x,y,c

              

                    
            
            
            
            
            
            

    list_temp1 = [] ###calculation the distance of bargaing solution Z with other voters not ones in Q excluding constant 1


    for Q_index in range(ncr(n,3)):
        for i in set(range(n))-set(mapping_3_subset_list[Q_index]):
            for S in powerset(set(range(n))-set({i})):
                for j in range(3): ##all possible orderings of Nash(a,b,c)- only position of c matters 
                    list_temp1.append(-2*Z[3*Q_index+j,mapping_all_subset_dict[frozenset(set(S)|set({i}))]])
    list_temp_const_1 = []
    list_temp_const_2 = []

    for Q_index in range(ncr(n,3)): ##calculating the constants 
        for i in set(range(n))-set(mapping_3_subset_list[Q_index]):
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
                for Q_prime in findsubsets(set(mapping_3_subset_list[j]),i):
                    for V_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                        list_temp3.append(X[mapping_all_subset_dict[frozenset(set(Q_prime)|set(V_prime))]])
        
        if (list_diff_R[j]=='0'): ## considers Case 1 as defined in every condition. 
            model.addConstr(gp.quicksum(list_temp3) >=1) ##puts desired constraint on X(xy) + X(xyc) + X(xc)+ X(yc) on a incremental space of x,y,c 
            for Q_prime in findsubsets(set(mapping_3_subset_list[j]),1): ### identifying the disagreement point
                    for U_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                      for l in range(3):
                        model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset(set(Q_prime)|set(U_prime))]]==0) ## Z(x)=Z(y)= Z(c) = 0 in the incremental budget space

                        if (list(mapping_3_subset_list[j])[(l+1)%3] in Q_prime): ##choosing one bargaining budget x
                          model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(Q_prime))|set(U_prime))]]==alpha_1[3*j+l]*X[mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(Q_prime))|set(U_prime))]]) ###Z(yc) = alpha*X(yc) in incremental budget space of $\{x,y,c\}$

                        elif (list(mapping_3_subset_list[j])[(l+2)%3] in Q_prime): ##choosing other bargaining y
                          model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(Q_prime))|set(U_prime))]]== alpha_2[3*j+l]*X[mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(Q_prime))|set(U_prime))]]) ###Z(xc) = beta*X(xc) in incremental budget space of $\{x,y,c\}$
                        
                        
            ##print("entered and finished part1",j)
        elif (list_diff_R[j]=='1'):
            ##print ("entering here",list_diff_R[j])
            model.addConstr(gp.quicksum(list_temp3) <=1)
            for Q_prime in findsubsets(set(mapping_3_subset_list[j]),2):
                    for U_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                      for l in range(3):
                        model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset(set(Q_prime)|set(U_prime))]]==X[mapping_all_subset_dict[frozenset(set(Q_prime)|set(U_prime))]]) ## Z(xy) = X(xy); Z(yc)= X(yc); Z(xc)=X(xc) = 0 in the incremental budget space


                        if (list(mapping_3_subset_list[j])[(l+1)%3] in Q_prime and list(mapping_3_subset_list[j])[l] in Q_prime): ##Q_prime is one pair of points ${x,c}$
                            model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(Q_prime))|set(U_prime))]]== alpha_1[3*j+l]*X[mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(Q_prime))|set(U_prime))]]) ###Z(y) = alpha*X(y)

                        elif (list(mapping_3_subset_list[j])[(l+2)%3] in Q_prime and list(mapping_3_subset_list[j])[l] in Q_prime): ##Q_prime is one pair of points ${y,c}$
                            model.addConstr(Z[3*j+l,mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(Q_prime))|set(U_prime))]]== alpha_2[3*j+l]*X[mapping_all_subset_dict[frozenset((mapping_3_subset_list[j]-set(Q_prime))|set(U_prime))]]) ###Z(x) = beta*X(x)

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
            
    if (model.ObjVal > current_max):
      current_max = max(model.objVal,current_max) ##checking maximum value obtained
      current_idx = itr

     

    
    if (current_max > 5e-5):
      print ("Not working case no",itr,"has optimum value", current_max)
    
print ("Final maximum obtained",current_max,"max_idx",current_idx)

          


