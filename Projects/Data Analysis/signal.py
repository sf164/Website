# %%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy.special import erf



def gaussian(x):
    return (1/np.sqrt(2*np.pi)*np.exp((-x**2)/2))

def bi_skewed_gauss(x, A, B, mean, mean2, alpha):
    phi = (1/2)*(1 + erf(alpha*((x - mean))/np.sqrt(2)))
    return A*2*gaussian((x - mean))*phi + B*gaussian(x - mean2)

x = np.array(np.linspace(0, 100, 10000))

def residuals(x, data, fit_data, unc):
    y = (data - fit_data)/(unc)
    plt.errorbar(x, y, yerr=unc, fmt = "k.", ecolor='red', label = 'error', lw = 1)
    plt.axhline(y = 0, color = 'black')
    plt.xlabel('Calibrated Energy (keV)')
    plt.ylabel('Events')
    plt.grid()
    plt.show()

#plt.plot(x, bi_skewed_gauss(x, 1, 30, 10, 20), label = 'fit')
plt.legend()
plt.grid()
plt.show()


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


with open(r"C:\Users\shuma\OneDrive\Desktop\WORK\Vscode\Fall 2024\PHY324\Data Analysis Assignment\signal_p3.pkl", "rb") as file:
    signal_data=pickle.load(file)

for n in range(1):
    plt.plot(signal_data['evt_%i'%n] - np.average(signal_data['evt_%i'%n][:1000]), alpha= 0.3)
plt.xlabel('Time (ms)')
plt.ylabel('Signal (V)')
plt.title('Signal data ')
plt.legend(loc = 1)
plt.show()

s_amp1 = np.zeros(1000)
s_area1 = np.zeros(1000)

for n in range(1000):
    current_data = signal_data['evt_%i'%n]
    s_amp1_calc = np.max(current_data) - np.average(current_data[:1000])
    s_amp1[n] = s_amp1_calc
s_amp1*= 1000


def energy_estimator_plot(amp, num_bins, bin_range, initial_g):
    n, bin_edges,patches = plt.hist(amp, bins=num_bins, range = bin_range, color = 'cornflowerblue', zorder= 2, rwidth = 0.9, label = 'Data')
    plt.xlabel('Amplitude: M2 (mV)')
    plt.ylabel('Events')
    plt.xlim(bin_range)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    sig = np.sqrt(n)
    sig = np.where(sig == 0, 3, sig)
    plt.errorbar(bin_centers, n , yerr = sig, fmt = 'none', ecolor = 'red', label= 'error')
    popt, pcov = curve_fit(myGauss, bin_centers, n, sigma = sig, p0 = initial_g, absolute_sigma= True)
    n_fit = myGauss(bin_centers, *popt)

    chisquared = np.sum(((n - n_fit)/sig)**2)
    dof = num_bins - len(popt)
    x_bestfit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    y_bestfit = myGauss(x_bestfit, *popt)

    fontsize = 14
    #plt.plot(x_bestfit, y_bestfit, color = 'purple', label='Fit')
    plt.legend(loc = 1)
    plt.savefig('pre_calibration signal amplitude.jpg')
    plt.show()


energy_estimator_plot(s_amp1, 30, [0.03, 0.4], (250, 0.022, 0.01, 10))

def calibration(amp, num_bins, bin_range, method, initial_g=()):
    n, bin_edges, _ = plt.hist(amp, num_bins, bin_range, color = 'cornflowerblue', zorder = 2, rwidth = 0.9, label = 'Data')
    plt.xlabel(f'Calibrated Energy (keV)')
    plt.ylabel('Events')
    plt.xlim(bin_range)

    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    sig = np.sqrt(n)
    sig=np.where(sig < 2.5, 2.5, sig)
    plt.errorbar(bin_centers, n, yerr=sig, fmt='none', ecolor='red', label="error")
    popt, pcov = curve_fit(bi_skewed_gauss, bin_centers, n, 
             sigma = sig, p0= initial_g, absolute_sigma=True)
    para_unc = np.sqrt(np.diag(pcov))
    print(*popt)
    print(*para_unc)
    n_fit = bi_skewed_gauss(bin_centers, *popt)
    chisquared = np.sum(((n - n_fit)/sig)**2)
    dof = num_bins - len(popt)
    x_bestfit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    y_bestfit = bi_skewed_gauss(x_bestfit, *popt)
    print('Calibration Data using the energy method of max - baseline average')
    plt.plot(x_bestfit, y_bestfit, label='Fit', color= 'purple')
    print(f'mu = {popt[1]}')
    #plt.text(0.11, 140, r'$\mu$ = %3.2f mV'%(popt[1]), fontsize=fontsize)
    #plt.text(0.11, 120, r'$\sigma$ = %3.2f mV'%(popt[2]), fontsize=fontsize)
    #plt.text(0.11, 100, r'$\chi^2$/DOF=', fontsize=fontsize)
    #plt.text(0.11, 80, r'%3.2f/%i'%(chisquared,dof), fontsize=fontsize)
    #plt.text(0.11, 60, r'$\chi^2$ prob.= %1.1f'%(1-chi2.cdf(chisquared,dof)), fontsize=fontsize)
    print(f'chi^2 = {chisquared/dof}' )
    plt.legend(loc=1)
    plt.show()
    residuals(bin_centers, n, n_fit, sig)
    
energy_s_amp1 = s_amp1*10/0.24 #calibration factor 2


calibration(energy_s_amp1, 25, [2, 10], 2, (175, 50, 2.6, 6.3, 120000))