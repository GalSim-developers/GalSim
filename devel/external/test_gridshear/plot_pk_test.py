import numpy as np
import matplotlib.pyplot as plt
import galsim

# read in theory quantities
tfile = 'ps.wmap7lcdm.2000.dat'
tdata = np.loadtxt(tfile)
tell = tdata[:,0]
tcell = tdata[:,1]
tdel = tcell*(tell**2)/(2.*np.pi)
tab = galsim.LookupTable(tell, tdel, x_log=True, f_log=True)

# first do P_E case
## read in data
galsim_file = 'output/ps.results.input_pe.dat'
galsim_data = np.loadtxt(galsim_file)
ell = galsim_data[:,0]
galsim_del_ee = (ell**2)*galsim_data[:,1]/(2.*np.pi)
galsim_del_bb = (ell**2)*galsim_data[:,2]/(2.*np.pi)
galsim_del_eb = (ell**2)*galsim_data[:,3]/(2.*np.pi)
sht_file = 'output/ps.results.input_pe.sht.dat'
sht_data = np.loadtxt(sht_file)
sht_del_ee = (ell**2)*sht_data[:,1]/(2.*np.pi)
sht_del_bb = (ell**2)*sht_data[:,2]/(2.*np.pi)
sht_del_eb = (ell**2)*sht_data[:,3]/(2.*np.pi)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tell, tdel, color='black', linestyle='dotted', label='theory')
ax.plot(ell, galsim_del_ee, color='red', label='GalSim EE')
ax.plot(ell, sht_del_ee, color='red', linestyle='dashed', label='SHT EE')
ax.plot(ell, galsim_del_bb, color='blue', label='GalSim BB')
ax.plot(ell, sht_del_bb, color='blue', linestyle='dashed', label='SHT BB')
#ax.plot(ell, galsim_del_eb, color='green', label='GalSim EB')
#ax.plot(ell, sht_del_eb, color='green', linestyle='dashed', label='SHT EB')
ax.set_xscale('log')
ax.set_yscale('log')
plt.ylim([1.e-7, 3.e-5])
plt.xlim([36.,1200.])
ax.set_xlabel('ell')
ax.set_ylabel('ell^2 Cell/2pi')
ax.set_title('Input P_E = P(k), P_B=0')
plt.legend(loc='upper left')
plt.savefig('output/compare_input_pe.eps')

ratio = galsim_del_ee/sht_del_ee
print ratio

galsim_file = 'output/ps.results.input_pe.fine.dat'
galsim_data = np.loadtxt(galsim_file)
ell = galsim_data[:,0]
galsim_del_ee = (ell**2)*galsim_data[:,1]/(2.*np.pi)
galsim_del_bb = (ell**2)*galsim_data[:,2]/(2.*np.pi)
galsim_del_eb = (ell**2)*galsim_data[:,3]/(2.*np.pi)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tell, tdel, color='black', linestyle='dotted', label='theory')
ax.plot(ell, galsim_del_ee, color='red', label='GalSim EE')
ax.plot(ell, galsim_del_bb, color='blue', label='GalSim BB')
ax.set_xscale('log')
ax.set_yscale('log')
plt.ylim([1.e-7, 3.e-5])
plt.xlim([36.,1200.])
ax.set_xlabel('ell')
ax.set_ylabel('ell^2 Cell/2pi')
ax.set_title('Input P_E = P(k), P_B=0')
plt.legend(loc='upper left')
plt.savefig('output/compare_input_pe.fine.eps')

ratio = galsim_del_ee
for ell_ind in range(len(ell)):
    ratio[ell_ind] /= tab(ell[ell_ind])
print ratio
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ell, ratio, label='Actual ratio')
const1 = np.zeros_like(ratio)+1.
ax.plot(ell, const1, label='Ideal ratio')
ax.set_xscale('log')
plt.ylim([0.8, 1.2])
plt.xlim([36.,1200.])
ax.set_xlabel('ell')
ax.set_ylabel('Ratio: GalSim EE vs. theory')
plt.legend(loc='upper right')
plt.savefig('output/compare_input_pe.fine.ratio.eps')


# next do P_B case
## read in data
galsim_file = 'output/ps.results.input_pb.dat'
galsim_data = np.loadtxt(galsim_file)
ell = galsim_data[:,0]
galsim_del_ee = (ell**2)*galsim_data[:,1]/(2.*np.pi)
galsim_del_bb = (ell**2)*galsim_data[:,2]/(2.*np.pi)
galsim_del_eb = (ell**2)*galsim_data[:,3]/(2.*np.pi)
sht_file = 'output/ps.results.input_pb.sht.dat'
sht_data = np.loadtxt(sht_file)
sht_del_ee = (ell**2)*sht_data[:,1]/(2.*np.pi)
sht_del_bb = (ell**2)*sht_data[:,2]/(2.*np.pi)
sht_del_eb = (ell**2)*sht_data[:,3]/(2.*np.pi)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tell, tdel, color='black', linestyle='dotted', label='theory')
ax.plot(ell, galsim_del_ee, color='red', label='GalSim EE')
ax.plot(ell, sht_del_ee, color='red', linestyle='dashed', label='SHT EE')
ax.plot(ell, galsim_del_bb, color='blue', label='GalSim BB')
ax.plot(ell, sht_del_bb, color='blue', linestyle='dashed', label='SHT BB')
#ax.plot(ell, galsim_del_eb, color='green', label='GalSim EB')
#ax.plot(ell, sht_del_eb, color='green', linestyle='dashed', label='SHT EB')
ax.set_xscale('log')
ax.set_yscale('log')
plt.ylim([1.e-7, 3.e-5])
plt.xlim([36.,1200.])
ax.set_xlabel('ell')
ax.set_ylabel('ell^2 Cell/2pi')
ax.set_title('Input P_E = 0, P_B=P(k)')
plt.legend(loc='upper left')
plt.savefig('output/compare_input_pb.eps')

# first do P_E case
## read in data
galsim_file = 'output/ps.results.input_peb.dat'
galsim_data = np.loadtxt(galsim_file)
ell = galsim_data[:,0]
galsim_del_ee = (ell**2)*galsim_data[:,1]/(2.*np.pi)
galsim_del_bb = (ell**2)*galsim_data[:,2]/(2.*np.pi)
galsim_del_eb = (ell**2)*galsim_data[:,3]/(2.*np.pi)
sht_file = 'output/ps.results.input_peb.sht.dat'
sht_data = np.loadtxt(sht_file)
sht_del_ee = (ell**2)*sht_data[:,1]/(2.*np.pi)
sht_del_bb = (ell**2)*sht_data[:,2]/(2.*np.pi)
sht_del_eb = (ell**2)*sht_data[:,3]/(2.*np.pi)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tell, tdel, color='black', linestyle='dotted', label='theory')
ax.plot(ell, galsim_del_ee, color='red', label='GalSim EE')
ax.plot(ell, sht_del_ee, color='red', linestyle='dashed', label='SHT EE')
ax.plot(ell, galsim_del_bb, color='blue', label='GalSim BB')
ax.plot(ell, sht_del_bb, color='blue', linestyle='dashed', label='SHT BB')
#ax.plot(ell, galsim_del_eb, color='green', label='GalSim EB')
#ax.plot(ell, sht_del_eb, color='green', linestyle='dashed', label='SHT EB')
ax.set_xscale('log')
ax.set_yscale('log')
plt.ylim([1.e-7, 3.e-5])
plt.xlim([36.,1200.])
ax.set_xlabel('ell')
ax.set_ylabel('ell^2 Cell/2pi')
ax.set_title('Input P_E = P_B = P(k)')
plt.legend(loc='upper left')
plt.savefig('output/compare_input_peb.eps')
