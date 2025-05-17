import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}
rc('font', **font)
# This changes the fonts for all graphs to make them bigger.



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

with open(r"C:\Users\shuma\OneDrive\Desktop\WORK\Vscode\Fall 2024\PHY324\Data Analysis Assignment\calibration_p3.pkl", "rb") as file:
    calibration_data=pickle.load(file)

pulse_template = pulse_shape(20, 80)
amplitude = np.zeros(1000)

x = np.linspace(0, 4096, 4096)
for n in range(1000):
    current_data = calibration_data['evt_%i'%n]
    popt, pcov = curve_fit(fit_pulse, x, current_data, p0 = (0.0002))
    amplitude[n] = popt[0]

amplitude*=1000 # convert from V to mV
num_bins=40 
bin_range=(0.1,0.42)

n, bin_edges, _ = plt.hist(amplitude, bins = num_bins, range = bin_range, color = 'cornflowerblue', zorder = 2, rwidth=0.9,  label = 'Data')
plt.xlabel('Amplitude (mV)')
plt.ylabel('Event')
plt.xlim(bin_range)



bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
sig = np.sqrt(n)
sig = np.where(sig==0, 3, sig)
plt.errorbar(bin_centers, n, yerr = sig, fmt = 'none', ecolor = 'red', label= 'error')


popt1, pcov1 = curve_fit(myGauss, bin_centers, n, 
             sigma = sig, p0=(100,0.25,0.05,5), absolute_sigma=True)
n1_fit = myGauss(bin_centers, *popt1)


chisquared1 = np.sum( ((n - n1_fit)/sig )**2)
dof1 = num_bins - len(popt1)

x_bestfit1 = np.linspace(bin_edges[0], bin_edges[-1], 1000)
y_bestfit1 = myGauss(x_bestfit1, *popt1) 

m1_para_unc = np.sqrt(np.diag(pcov1))

plt.plot(x_bestfit1, y_bestfit1, label = 'fit', color = 'purple')
print(*popt1)
print(*m1_para_unc)
print(f'$\chi^2 = ${chisquared1/dof1}')
plt.legend(loc =1)
plt.show()

def residuals(x, data, fit_data, unc):
    y = (data - fit_data)/(unc)
    plt.errorbar(x, y, yerr=unc, fmt = "k.", ecolor='red', label = 'error', lw = 1)
    plt.axhline(y = 0, color = 'black')
    plt.xlabel('Calibrated Energy (keV)')
    plt.ylabel('Events')
    plt.grid()
    plt.show()
residuals(bin_centers, n, n1_fit, sig)

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
    residuals(bin_centers, n, n_fit, sig)
    print(f'$\chi^2 = {chisquared/dof}$')

energy_amp = amplitude*10/0.21
nbin_range = [3, 18]
initial = (150, 10, 2, 0)
calibration(energy_amp, 45, nbin_range, 2,  initial)

