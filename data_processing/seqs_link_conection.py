# SUMMARY: we are going to convert the txt file into the npy file in which is going to be represented the peaks values for each sequence and patient
# IMPUT: We introduce a txt file with a matrix with the peaks values and a header
# OUTPUT: A npy file in which is the peaks values codifing as we need for the execution of the AI-TAC model

#MODULES
import numpy as np
import sys

####################################################################################################

dic_imp_out_file = sys.argv[1]

n_seqs = 215920 + 1

links = ["b'BRCA_" + str(i) + ".peak_" + str(i) + "'" for i in range(1:n_seqs)]

np.save(dic_imp_out_file  + '/' + 'peak_names.npy', links)
