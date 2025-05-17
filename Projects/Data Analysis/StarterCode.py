# %%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

from scipy.special import erf

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}
rc('font', **font)
# This changes the fonts for all graphs to make them bigger.

def residuals(x, data, fit_data, unc):
    y = (data - fit_data)/(unc)
    plt.errorbar(x, y, yerr=unc, fmt = "k.", ecolor='red', label = 'error', lw = 1)
    plt.axhline(y = 0, color = 'black')
    plt.xlabel('Calibrated Energy (keV)')
    plt.ylabel('Events')
    plt.grid()
    plt.show()



def myGauss(x, A, mean, width, base):
    return A*np.exp(-(x-mean)**2/(2*width**2)) + base
# This is my fitting function, a Guassian with a uniform background.

def pulse_shape(t_rise, t_fall):
    xx=np.linspace(0, 4095, 4096)
    yy = -(np.exp(-(xx-1000)/t_rise)-np.exp(-(xx-1000)/t_fall))
    yy[:1000]=0
    yy /= np.max(yy)
    return yy

def fit_pulse(x, A):
    _pulse_template = pulse_shape(20,80)
    xx=np.linspace(0, 4095, 4096)
    return A*np.interp(x, xx, _pulse_template)
# fit_pulse can be used by curve_fit to fit a pulse to the pulse_shape

with open(r"C:\Users\shuma\OneDrive\Desktop\WORK\Vscode\Fall 2024\PHY324\Data Analysis Assignment\calibration_p3.pkl", "rb") as file:
    calibration_data=pickle.load(file)

pulse_template = pulse_shape(20,80)
plt.plot(pulse_template/2000, label='Pulse Template', color='r')
for itrace in range(1):
    plt.plot(calibration_data['evt_%i'%itrace] -np.average(calibration_data['evt_%i'%itrace][:1000]) , alpha=0.3)
plt.xlabel('Time (ms)')
plt.ylabel('Pulse Readout (V)')
plt.legend(loc=1)
plt.show()
""" 
This shows the first 10 data sets on top of each other.
Always a good idea to look at some of your data before analysing it!
It also plots our pulse template which has been scaled to be slightly 
larger than any of the actual pulses to make it visible.
"""




amp1=np.zeros(1000)
amp2=np.zeros(1000)
amp3=np.zeros(1000)
amp4=np.zeros(1000)
amp5=np.zeros(1000)
amp6=np.zeros(1000)
area1=np.zeros(1000)
area2=np.zeros(1000)
area3=np.zeros(1000)
area4=np.zeros(1000)
area5=np.zeros(1000)
area6=np.zeros(1000)

pulse_fit=np.zeros(1000)
# These are the 6 energy estimators as empty arrays of the correct size.

for ievt in range(1000):
    current_data = calibration_data['evt_%i'%ievt]
    amp1_calculation = np.max(current_data) - np.min(current_data)
    amp1[ievt] = amp1_calculation

amp1*=1000 # convert from V to mV
num_bins1=40 
bin_range1=(0.1,0.5)
"""
These two values were picked by trial and error. You'll 
likely want different values for each estimator.
"""

"""
def trim_energy(arr, min, max=10):
    arr2 = []
    for i in arr:
        if min <= i <= max:
            arr2.append(i)
    return np.array(arr2)
amp1_f = trim_energy(amp1, 0.1)
"""


n1, bin_edges1, _ = plt.hist(amp1, bins=num_bins1, range=bin_range1, color='cornflowerblue', zorder=2, rwidth=0.9, label='Data')
# This plots the histogram AND saves the counts and bin_edges for later use

plt.xlabel('Amplitude: (mV)')
plt.ylabel('Events')
plt.xlim(bin_range1)  
# If the legend covers some data, increase the plt.xlim value, maybe (0,0.5)

bin_centers1 = 0.5*(bin_edges1[1:]+bin_edges1[:-1])
"""
This gives us the x-data which are the centres of each bin.
This is visually better for plotting errorbars.
More important, it's the correct thing to do for fitting the
Gaussian to our histogram.
It also fixes the shape -- len(n1) < len(bin_edges1) so we
cannot use 
plt.plot(n1, bin_edges1)
as it will give us a shape error.
"""

sig1 = np.sqrt(n1)
sig1=np.where(sig1==0, 3, sig1) 
# The uncertainty on 0 count is 1, not 0. Replace all 0s with 1s.

plt.errorbar(bin_centers1, n1, yerr=sig1, fmt='none', ecolor='red', label="error")
# This adds errorbars to the histograms, where each uncertainty is sqrt(y)

popt1, pcov1 = curve_fit(myGauss, bin_centers1, n1, 
             sigma = sig1, p0=(100,0.25,0.05,5), absolute_sigma=True)
n1_fit = myGauss(bin_centers1, *popt1)
"""
n1_fit is our best fit line using our data points.
Note that if you have few enough bins, this best fit
line will have visible bends which look bad, so you
should not plot n1_fit directly. See below.
"""

chisquared1 = np.sum( ((n1 - n1_fit)/sig1 )**2)
dof1 = num_bins1 - len(popt1)
# Number of degrees of freedom is the number of data points less the number of fitted parameters

x_bestfit1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
y_bestfit1 = myGauss(x_bestfit1, *popt1) 
# Best fit line smoothed with 1000 datapoints. Don't use best fit lines with 5 or 10 data points!

m1_para_unc = np.sqrt(np.diag(pcov1))

fontsize=14
plt.plot(x_bestfit1, y_bestfit1, label='Fit', color= 'purple')
print(f'$\mu$ = {popt1[1]} +- {m1_para_unc[1]}')
print('method 1')
#plt.text(0.11, 140, r'$\mu$ = %3.2f mV'%(popt1[1]), fontsize=fontsize)
print(f'$\sigma$ = {popt1[2]} +- {m1_para_unc[2]}')
#plt.text(0.11, 120, r'$\sigma$ = %3.2f mV'%(popt1[2]), fontsize=fontsize)
print(f'$\chi^2$ = {chisquared1/dof1}')
print(*popt1)
print(*m1_para_unc)
#plt.text(0.11, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
#plt.text(0.11, 80, r'%3.2f/%i'%(chisquared1,dof1), fontsize=fontsize)
#plt.text(0.11, 60, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared1,dof1)), fontsize=fontsize)
plt.legend(loc=1)
plt.show()
residuals(bin_centers1, n1, n1_fit, sig1)

"""
    For amp2 - Maximum value minus the baseline average
"""
for n in range(1000):
    current_data = calibration_data['evt_%i'%n]
    amp2_calc = np.max(current_data) - np.average(current_data[:1000])
    amp2[n] = amp2_calc 

amp2*=1000 #converting V to mV
num_bins2=40
bin_range2=(0.1,0.4)


n2, bin_edges2, patches = plt.hist(amp2, bins=num_bins2, range=bin_range2, color='cornflowerblue', zorder=2, rwidth=0.9, label='Data')
plt.xlabel('Amplitude (mV)')
plt.ylabel('Events')
plt.xlim(bin_range2)

bin_centers2 = 0.5*(bin_edges2[1:]+ bin_edges2[:-1])



sig2 = np.sqrt(n2)
sig2 = np.where(sig2 == 0, 3, sig2)

plt.errorbar(bin_centers2, n2, yerr= sig2, fmt='none', ecolor='red', label='error')

popt2, pcov2 = curve_fit(myGauss, bin_centers2, n2, 
             sigma = sig2, p0=(100,0.25,0.05,5), absolute_sigma=True)
n2_fit = myGauss(bin_centers2, *popt2)

chisquared2 = np.sum( ((n2 - n2_fit)/sig2 )**2)
dof2 = num_bins2 - len(popt2)

x_bestfit2 = np.linspace(bin_edges2[0], bin_edges2[-1], 1000)
y_bestfit2 = myGauss(x_bestfit2, *popt2) 

m2_para_unc = np.sqrt(np.diag(pcov2))

fontsize=14
plt.plot(x_bestfit2, y_bestfit2, color = 'purple', label='Fit')
print(f'$\mu$ = {popt2[1]} +- {m2_para_unc[1]}')
print('Method 2')
print(f'$\sigma$ = {popt2[2]} +- {m2_para_unc[2]}')
print(f'$\chi^2$ = {chisquared2/dof2}')
print(*popt2)
print(*m2_para_unc)
#plt.text(0.11, 140, r'$\mu$ = %3.2f mV'%(popt2[1]), fontsize=fontsize)
#plt.text(0.11, 120, r'$\sigma$ = %3.2f mV'%(popt2[2]), fontsize=fontsize)
#plt.text(0.11, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
#plt.text(0.11, 80, r'%3.2f/%i'%(chisquared2,dof2), fontsize=fontsize)
#plt.text(0.11, 60, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared2,dof2)), fontsize=fontsize)
plt.legend(loc=1)
plt.show()
residuals(bin_centers2, n2, n2_fit, sig2)



"""
    For amp3 - Sum of all values
"""
for n in range(1000):
    current_data = calibration_data['evt_%i'%n]
    amp3_calc = np.sum(current_data)
    amp3[n] = amp3_calc

amp3*=1000 #converting V to mV
num_bins3=30
bin_range3=(-100,160)


n3, bin_edges3, patches = plt.hist(amp3, bins=num_bins3, range=bin_range3,color='cornflowerblue', zorder=2, rwidth=0.9, label='Data')
plt.xlabel('Amplitude (mV)')
plt.ylabel('Events')
plt.xlim(bin_range3)

bin_centers3 = 0.5*(bin_edges3[1:]+ bin_edges3[:-1])


sig3 = np.sqrt(n3)
sig3 = np.where(sig3 == 0, 3, sig3)

plt.errorbar(bin_centers3, n3, yerr= sig3, fmt='none', ecolor='red', label='error')

popt3, pcov3 = curve_fit(myGauss, bin_centers3, n3, 
             sigma = sig3, p0=(100,40,80,5), absolute_sigma=True)
n3_fit = myGauss(bin_centers3, *popt3)

chisquared3 = np.sum( ((n3 - n3_fit)/sig3 )**2)
dof3 = num_bins3 - len(popt3)

x_bestfit3 = np.linspace(bin_edges3[0], bin_edges3[-1], 1000)
y_bestfit3 = myGauss(x_bestfit3, *popt3) 

m3_para_unc = np.sqrt(np.diag(pcov3))
fontsize=14
plt.plot(x_bestfit3, y_bestfit3, color = 'purple',label='Fit')
print(f'$\mu$ = {popt3[1]} +- {m3_para_unc[1]}')
print('Method 3')
print(f'$\sigma$ = {popt3[2]} +- {m3_para_unc[2]}')
print(f'$\chi^2$ = {chisquared3/dof3}')
print(*popt3)
print(*m3_para_unc)
#plt.text(-95, 80, r'$\mu$ = %3.2f mV'%(popt3[1]), fontsize=fontsize)
#plt.text(-95, 70, r'$\sigma$ = %3.2f mV'%(popt3[2]), fontsize=fontsize)
#plt.text(-95, 60, r'$\chi^2$/DOF=', fontsize=fontsize)
#plt.text(-95, 50, r'%3.2f/%i'%(chisquared2,dof3), fontsize=fontsize)
#plt.text(-95, 40, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared2,dof3)), fontsize=fontsize)
plt.legend(loc=1)
plt.show()

residuals(bin_centers3, n3, n3_fit, sig3)


"""
    For amp4 - Sum of all (values - baseline average)
"""
for n in range(1000):
    current_data = calibration_data['evt_%i'%n]
    baseline_avg = np.average(current_data[:1000])
    amp4_calc = np.sum(current_data - baseline_avg) 
    amp4[n] = amp4_calc

amp4*=1000 #converting V to mV
num_bins4=35

bin_range4=(-15,70)

n4, bin_edges4, patches = plt.hist(amp4, bins=num_bins4, range=bin_range4, color='cornflowerblue', zorder=2, rwidth=0.9, label='Data')
plt.xlabel('Amplitude (mV)')
plt.ylabel('Events')
plt.xlim(bin_range4)

bin_centers4 = 0.5*(bin_edges4[1:]+ bin_edges4[:-1])


sig4 = np.sqrt(n4)
sig4 = np.where(sig4 == 0, 3, sig4)


plt.errorbar(bin_centers4, n4, yerr= sig4, fmt='none', ecolor='red', label='error')

popt4, pcov4 = curve_fit(myGauss, bin_centers4, n4, 
             sigma = sig4, p0=(140,30,20,5), absolute_sigma=True)
n4_fit = myGauss(bin_centers4, *popt4)

chisquared4 = np.sum( ((n4 - n4_fit)/sig4 )**2)
dof4 = num_bins4 - len(popt4)

x_bestfit4 = np.linspace(bin_edges4[0], bin_edges4[-1], 1000)
y_bestfit4 = myGauss(x_bestfit4, *popt4) 

m4_para_unc = np.sqrt(np.diag(pcov4))

fontsize=14
print(f'$\mu$ = {popt4[1]} +- {m4_para_unc[1]}')
print('Method 4')
print(f'$\sigma$ = {popt4[2]} +- {m4_para_unc[2]}')
print(f'$\chi^2$ = {chisquared4/dof4}')
print(*popt4)
print(*m4_para_unc)
plt.plot(x_bestfit4, y_bestfit4, color = 'purple',label='Fit')
#plt.text(-13, 105, r'$\mu$ = %3.2f mV'%(popt4[1]), fontsize=fontsize)
#plt.text(-13, 90, r'$\sigma$ = %3.2f mV'%(popt4[2]), fontsize=fontsize)
#plt.text(-13, 75, r'$\chi^2$/DOF=', fontsize=fontsize)
#plt.text(-13, 60, r'%3.2f/%i'%(chisquared2,dof4), fontsize=fontsize)
#plt.text(-13, 45, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared2,dof4)), fontsize=fontsize)
plt.legend(loc=1)
plt.show()

residuals(bin_centers4, n4, n4_fit, sig4)


"""
    For amp5 - Sum of only the pulse
"""
for n in range(1000):
    current_data = calibration_data['evt_%i'%n]
    amp5_calc = np.sum(current_data[1000:])
    amp5[n] = amp5_calc


amp5*=1000 #converting V to mV
num_bins5=35


bin_range5=(-60,120)

n5, bin_edges5, patches = plt.hist(amp5, bins=num_bins5, range=bin_range5, color='cornflowerblue', zorder=2, rwidth=0.9, label='Data')
plt.xlabel('Ampltude (mV)')
plt.ylabel('Events')
plt.xlim(bin_range5)

bin_centers5 = 0.5*(bin_edges5[1:]+ bin_edges5[:-1])


sig5 = np.sqrt(n5)
sig5 = np.where(sig5 == 0, 3, sig5)

plt.errorbar(bin_centers5, n5, yerr= sig5, fmt='none', ecolor='red', label='error')

popt5, pcov5 = curve_fit(myGauss, bin_centers5, n5, 
             sigma = sig5, p0=(140,50,70,5), absolute_sigma=True)
n5_fit = myGauss(bin_centers5, *popt5)

chisquared5 = np.sum( ((n5 - n5_fit)/sig5 )**2)
dof5 = num_bins5 - len(popt5)

x_bestfit5 = np.linspace(bin_edges5[0], bin_edges5[-1], 1000)
y_bestfit5 = myGauss(x_bestfit5, *popt5) 
m5_para_unc = np.sqrt(np.diag(pcov5))

fontsize=14
plt.plot(x_bestfit5, y_bestfit5, color = 'purple', label='Fit')
print(f'$\mu$ = {popt5[1]}')
print('Method 5')
print(f'$\sigma$ = {popt5[2]}')
print(f'$\chi^2$ = {chisquared5/dof5}')
print(*popt5)
print(*m5_para_unc)
#plt.text(-54, 78, r'$\mu$ = %3.2f mV'%(popt5[1]), fontsize=fontsize)
#plt.text(-54, 68, r'$\sigma$ = %3.2f mV'%(popt5[2]), fontsize=fontsize)
#plt.text(-54, 58, r'$\chi^2$/DOF=', fontsize=fontsize)
#plt.text(-54, 48, r'%3.2f/%i'%(chisquared2,dof5), fontsize=fontsize)
#plt.text(-54, 38, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared2,dof5)), fontsize=fontsize)
plt.legend(loc=1)
plt.show()

residuals(bin_centers5, n5, n5_fit, sig5)

"""Method 6 - Fit method (fit the pulse using skewed gaussian and then)"""

def gaussian(x):
    return (1/np.sqrt(2*np.pi)*np.exp((-x**2)/2))

def bi_skewed_gauss(x, A, B, mean, mean2, alpha):
    phi = (1/2)*(1 + erf(alpha*((x - mean))/np.sqrt(2)))
    return A*2*gaussian((x - mean))*phi + B*gaussian(x - mean2)











"""
Calibration now? find a calibration factor such that the x-axis is in keV and the peak is at 10keV. energy_amp1 = amp_1*conversion_factor
"""
energy_amp1 = amp1*10/0.3
nbin_range1 = [3, 18]
m1_initial = (150, 10, 2, 0)

energy_amp2 = amp2*10/0.24
nnum_bins2 = 45
nbin_range2 = [5, 15]
m2_initial = (120, 10, 2, 0)

energy_amp3 = amp3*10/(27.9)
nbin_range3 = [-40, 60]
m3_initial = (100, 10, 50, 0) 

energy_amp4 = amp4*10/(27.4)
nbin_range4 = [-5, 25]
m4_initial = (450, 10, 20, 4)

energy_amp5 = amp5*10/(26.7)
nbin_range5 = [-25, 50]
m5_initial = (70, 10, 20, 0)




def calibration(amp, num_bins, bin_range, method, initial_g=()):
    n, bin_edges, _ = plt.hist(amp, num_bins, bin_range, color = 'cornflowerblue', zorder = 2, rwidth = 0.9, label = 'Data')
    plt.xlabel(f'Calibrated Energy: (keV)')
    plt.ylabel('Events')
    plt.xlim(bin_range)

    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    sig = np.sqrt(n)
    sig=np.where(sig < 2.5, 2.5, sig)
    plt.errorbar(bin_centers, n, yerr=sig, fmt='none', ecolor='red', label="error")
    popt, pcov = curve_fit(myGauss, bin_centers, n, 
             sigma = sig, p0=initial_g, absolute_sigma=True)
    para_unc = np.sqrt(np.diag(pcov))
    n_fit = myGauss(bin_centers, *popt)
    chisquared = np.sum(((n - n_fit)/sig)**2)
    dof = num_bins - len(popt)
    x_bestfit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    y_bestfit = myGauss(x_bestfit, *popt)
    print(*popt)
    print(*para_unc)
    plt.plot(x_bestfit, y_bestfit, label='Fit', color= 'purple')
    #plt.text(0.11, 140, r'$\mu$ = %3.2f mV'%(popt[1]), fontsize=fontsize)
    #plt.text(0.11, 120, r'$\sigma$ = %3.2f mV'%(popt[2]), fontsize=fontsize)
    #plt.text(0.11, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
    #plt.text(0.11, 80, r'%3.2f/%i'%(chisquared,dof), fontsize=fontsize)
    #plt.text(0.11, 60, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared,dof)), fontsize=fontsize)
    plt.legend(loc=1)
    plt.show()
    print(f'$\chi^2 = {chisquared/dof}$')
    residuals(bin_centers, n, n_fit, sig)



# firts method calibrated
calibration(energy_amp1, num_bins1, nbin_range1, 1, m1_initial)
calibration(energy_amp2, nnum_bins2, nbin_range2, 2, m2_initial)
calibration(energy_amp3, num_bins3, nbin_range3, 3, m3_initial)
calibration(energy_amp4, num_bins4, nbin_range4, 4, m4_initial)
calibration(energy_amp5, num_bins5, nbin_range5, 5, m5_initial)

#residuals(bin_centers2, n2, n2_fit, sig2)


"""
Look how bad that chi-squared value (and associated probability) is!
If you look closely, the first 5 data points (on the left) are
responsible for about half of the chi-squared value. It might be
worth excluding them from the fit and subsequent plot.

Now your task is to find the calibration factor which converts the
x-axis of this histogram from mV to keV such that the peak (mu) is 
by definition at 10 keV. You do this by scaling each estimator (i.e.
the values of amp1) by a multiplicative constant with units mV / keV.
Something like:

energy_amp1 = amp1 * conversion_factor1

where you have to find the conversion_factor1 value. Then replot and
refit the histogram using energy_amp1 instead of amp1. 
If you do it correctly, the new mu value will be 10 keV, and the new 
sigma value will be the energy resolution of this energy estimator.

Note: you should show this before/after conversion for your first
energy estimator. To save space, only show the after histograms for
the remaining 5 energy estimators.
"""