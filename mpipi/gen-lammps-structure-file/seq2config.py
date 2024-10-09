import numpy as np
from sys import argv

'''
For details on using this script, please consult ReadMe.txt
'''

if len(argv) > 1:

	script = argv[0]
	seqType = int(argv[1])

	if seqType == 3:
		seq = str(argv[2])
		nres = int(len(seq)/3)
		charges = np.loadtxt('aa_charges_three_letters.txt', dtype=str)
		types = np.loadtxt('aa_types_three_letters.txt', dtype=str)

		typescharges = []

		count = 0
		for i in range(nres):
			k, = np.where(charges[:,0] == seq[count:count+3])
			p, = np.where(types[:,0] == seq[count:count+3])
			count += 3
			typescharges.append([int(types[p,1]),float(charges[k,1])])

	elif seqType == 1:
		seq = str(argv[2])
		nres = len(seq)
		charges = np.loadtxt('aa_charges_one_letter.txt', dtype=str)
		types = np.loadtxt('aa_types_one_letter.txt', dtype=str)

		typescharges = []

		for i in range(nres):
			k, = np.where(charges[:,0] == seq [i])
			p, = np.where(types[:,0] == seq [i])
			typescharges.append([int(types[p,1]),float(charges[k,1])])

else:
    seq = np.loadtxt('seq.txt', dtype=str)
    seqType = len(seq[0])

    if seqType == 3:
        charges = np.loadtxt('aa_charges_three_letters.txt', dtype=str)
        types = np.loadtxt('aa_types_three_letters.txt', dtype=str)
        nres = len(seq)
        
        typescharges = []

        for i in range(nres):
            k, = np.where(charges[:,0] == seq[i])
            p, = np.where(types[:,0] == seq[i])
            typescharges.append([int(types[p,1]),float(charges[k,1])])

    elif seqType == 1:
        charges = np.loadtxt('aa_charges_one_letter.txt', dtype=str)
        types = np.loadtxt('aa_types_one_letter.txt', dtype=str)
        nres = len(seq)
        
        typescharges = []

        for i in range(nres):
            k, = np.where(charges[:,0] == seq [i] )
            p, = np.where(types[:,0] == seq [i] )
            typescharges.append([int(types[p,1]),float(charges[k,1])])

total_atoms = nres

no_atom_types = 40

out = open('myconfig.dat','w')
out.write('LAMMPS data file for IDP\n\n')
out.write('%d atoms\n'%total_atoms)
out.write('%d bonds\n\n'%(total_atoms-1))
out.write('%d atom types\n'%no_atom_types)
out.write('1 bond types\n\n')

min_box = nres * -3
max_box = nres * 3

out.write('%5f   %5f  xlo xhi\n'%(min_box,max_box))
out.write('%5f   %5f  ylo yhi\n'%(min_box,max_box))
out.write('%5f   %5f  zlo zhi\n\n'%(min_box,max_box))

out.write('Masses\n\n')

mass_dict = {1:   131.199997,    
             2:   57.0499992,    
             3:   128.199997,    
             4:   101.099998,    
             5:   156.199997,    
             6:   71.0800018,    
             7:   115.099998,    
             8:   129.100006,    
             9:   163.199997,    
             10:   99.0699997,    
             11:   113.199997,    
             12:   128.100006,    
             13:   186.199997,    
             14:   147.199997,    
             15:   87.0800018,    
             16:   137.100006,    
             17:   114.099998,    
             18:   97.1200027,    
             19:   103.099998,    
             20:   113.199997,    
             21:   131.199997,    
             22:    57.0499992,    
             23:    128.199997,    
             24:    101.099998,    
             25:   156.199997,    
             26:    71.0800018,    
             27:    115.099998,    
             28:   129.100006,    
             29:    163.199997,    
             30:    99.0699997,    
             31:    113.199997,    
             32:    128.100006,    
             33:    186.199997,    
             34:   147.199997,    
             35:    87.0800018,    
             36:    137.100006,    
             37:    114.099998,    
             38:    97.1200027,    
             39:   103.099998,   
             40:   113.199997
}

count = 1
while count <= no_atom_types:
    out.write('   %d %f\n' %(count,float(mass_dict[count])))
    count += 1

out.write('\nAtoms\n\n')


xcoord=0.0
for k,v in enumerate(typescharges):
    dis = k
    atom = k+1
    type = v[0]
    charge = v[1]

    if int(dis) < int(50):
        xcoord = xcoord + 6.0
        ycoord = 0
        zcoord = 0
    elif int(dis) >= int(50) and int(dis) < int(100):
        xcoord = xcoord
        ycoord = ycoord + 6.0
        zcoord = 0

    elif int(dis) >= int(100):
        xcoord = xcoord
        ycoord = ycoord
        zcoord = zcoord + 6.0

    out.write('%d 1 %d  %f %f  %f  %f\n' %(atom,type,charge,xcoord,ycoord,zcoord))

out.write('\nBonds\n\n')

atom_id = 1
bond_id = 1

while atom_id < total_atoms:
    out.write('%d 1 %d %d\n' %(bond_id,atom_id,atom_id+1))
    atom_id +=1
    bond_id +=1

out.close()