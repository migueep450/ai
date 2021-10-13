# SUMMARY: we are going to convert the txt file into the npy file in which is going to be represented the peaks values for each sequence and patient
# IMPUT: We introduce a txt file with a matrix with the peaks values and a header
# OUTPUT: A npy file in which is the peaks values codifing as we need for the execution of the AI-TAC model

#MODULES
import numpy as np
import sys

####################################################################################################

name_peak_values = sys.argv[1]
dic_imp_out_file = sys.argv[2]

f_peak_values = open(dic_imp_out_file + '/' + name_peak_values, 'rt')

f_peak_values.readline()
line = f_peak_values.readline()

peaks = []

while line:
	line = line [0:-2]
	peaks.append(line.split("\t")) # we add to the list 'peaks' the peaks values of one sequence for all the patients
	line = f_peak_values.readline()

np.save(dic_imp_out_file  + '/' + 'cell_type_array.npy', peaks)


f_peak_values.close()

