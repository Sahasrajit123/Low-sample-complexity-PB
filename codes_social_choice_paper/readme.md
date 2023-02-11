We have three files in this folder.

a) Toggle_permutations.py This file first chooses only one from the pair of cases which are obtained by toggling all 20 bits and then removes cases which may not be unique upto the permutations of the voters.

b) permutations.py This file only removes the cases which are not unique upto permutations of the voter prefernces.

c) participatory_budget_n_6_randomised_nashgurobi_upto_toggle_opt_formulation: This file solves the optimization problem described in Lemma 12 for all the cases obtained from toggle_permutations.py

d) participatory_budget_n_6_gurobi_median_opt_formulation.py: his file solves the optimization problem described in Lemma 8 for all the cases obtained from permutations.py 