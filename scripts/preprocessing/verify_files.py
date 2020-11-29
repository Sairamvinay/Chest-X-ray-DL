import os,pickle

LL = []
sums = 0
all_files = os.listdir('all_PA_files/')
all_files = ['all_PA_files/' + str(file) for file in all_files if file[-4:] == '.pkl']
print(all_files, " these files are there")
for file in all_files:
	print()
	with open(file,'rb') as f:
		L = pickle.load(f)
		print(len(set(L))," for filename: ",file)
		sums += len(set(L.keys()))
		LL += list(set(L.keys()))

assert(sums == len(list(LL)))
print(len(list(LL))," many files in total")
