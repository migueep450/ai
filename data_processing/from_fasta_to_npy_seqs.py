# SUMMARY: we are going to convert the fasta file into the npy file in which is going to be represented the sequences as we need for the execution of the model
# IMPUT: We introduce a fasta file with the names of the sequences and they are in the same order as the peak's values file
# OUTPUT: A npy file in which is the sequences codifing as we need for the execution of the AI-TAC model

#MODULES
import numpy as np
import sys



# FUNCTIONS(): - 
def one_hot_encoder(sequence):
	'''
	takes DNA sequence, outputs one-hot-encoded matrix with rows A, T, G, C
	'''
	l = len(sequence)
	codified_seq = np.zeros((4,l),dtype = 'int8')
	for j, i in enumerate(sequence):
		if i == "A" or i == "a":
			codified_seq[0][j] = 1
		elif i == "T" or i == "t":
			codified_seq[1][j] = 1
		elif i == "G" or i == "g":
			codified_seq[2][j] = 1
		elif i == "C" or i == "c":
			codified_seq[3][j] = 1
		else:
			return "contains_N"
	return codified_seq



###################################################################################################################################

name_fasta_file = sys.argv[1]
dic_imp_out_file = sys.argv[2]

fasta_f = open(dic_imp_out_file + '/' + name_fasta_file, 'rt')

fasta_f.readline() # Name of sequence
line = fasta_f.readline() # Sequence

sequencies = []

while line:
	sequencies.append(one_hot_encoder(line[0:-2]))
	fasta_f.readline() # Name of sequence
	line = fasta_f.readline() # Sequence

np.save(dic_imp_out_file  + '/' + 'one_hot_seqs.npy', np.stack(sequencies))

fasta_f.close()
