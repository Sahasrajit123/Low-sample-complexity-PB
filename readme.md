We have four files in this folder contains scripts for the paper "Low Sample Complexity Participatory Budgeting". Have to download 4 files together in a folder to run them.

a) Toggle_permutations.py 

This file generates all possible binary strings of length (20). Note that each of these strings indices the optimization problem formulation as described in Lemma 13 and 9. Then it performs two reductions in sequence. '0' at an index denotes the 3-sized subset $Q$ corresponding to that index belongs in Case-1 and '1' denotes the same for Case-2.

i) From every pairs of such strings which are toggles of each other, exactly one of them is chosen for the next step. (Follows from the result in Lemma 21). 

ii) From the remaining strings(each denoting a different case), it removes those cases which may not be unique after permutations of the voters (Described in the proof of Lemma 13)  


b) permutations.py This file only removes the cases which are not unique upto permutations of the voter prefernces. (Described in the proof of Lemma 9). Also note that cases in these files denote binary strings of length 20. 

Also note that in both "Toggle_permutations.py" and "permutations.py", the strings are returned as decimal representation of the binary sequence for easier representation.


c) participatory_budget_n_6_randomised_nashgurobi_upto_toggle_opt_formulation: This file solves the optimization problem described in Lemma 12 for all the cases obtained from toggle_permutations.py. These cases are directly passed to this file using the list "non_perm_upto_toggle" in the code.

d) participatory_budget_n_6_gurobi_median_opt_formulation.py: his file solves the optimization problem described in Lemma 8 for all the cases obtained from permutations.py These cases are directly passed to this file using the list "non_perm_list" in the code.
