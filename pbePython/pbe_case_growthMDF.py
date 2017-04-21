import matplotlib.pyplot as plt
from numpy import *
from scipy import integrate
import pbe_functions_solver as pbeslv

def setInitialCondition():
	
	# UNIFORM INITIAL CONDITION
	a = 10.0
	b = 20.0
	Npts = 400
	xspan = linspace(0.0, 100.0, Npts)
	n0_fnc = lambda v, a, b: 0 if v<a else (1/(b-a) if v<b else 0)
	n_t0_uni = empty(Npts)
	for i in arange(0, Npts):
		n_t0_uni[i] = n0_fnc(xspan[i], a, b)
	fig = plt.figure(figsize=(16,6))
	plt.subplot(1,2,1)
	plt.plot(xspan, n_t0_uni, 'kd-', lw = 2, label='I.C. uniform')
	plt.xlabel('x')
	plt.ylabel('n(l,t)')
	plt.legend()
	uniax = fig.gca() 

	# GAUSSIAN INITIAL CONDITION
	mu_ic = 20.0
	sigma_ic = 5.0
	n0_fnc = lambda v, mu_ic, sigma_ic: 1./(sigma_ic*sqrt(2*pi)) * exp(-1./2.*((v-mu_ic)/sigma_ic)**2)
	n_t0_gau = empty(Npts)
	for i in arange(0, Npts):
		n_t0_gau[i] = n0_fnc(xspan[i], mu_ic, sigma_ic)
	plt.subplot(1,2,2)
	plt.plot(xspan, n_t0_gau, 'kd-', lw = 2, label='I.C. gaussian')
	plt.xlabel('x')
	plt.ylabel('n(l,t)')
	plt.legend()
	plt.subplots_adjust(wspace = 0.4)
	gauax = fig.gca()
	fig.savefig('initialconditions.svg', transparent=False)

	# DEFINE DICTIONARY AS PARAMETER FOR SOLVER FUNCTIONS
	arg1 = {}
	arg1['Npts'] = Npts
	arg1['delta'] = xspan[1]-xspan[0]
	arg1['x'] = xspan
	growth_fnc = lambda xi: 1e3
	arg1['growth_fnc'] = growth_fnc
	diff_growth_fnc = lambda xi: 0.0
	arg1['diff_growth_fnc'] = diff_growth_fnc


	return arg1, n_t0_uni, n_t0_gau, uniax, gauax

def main():

	# UNIFORM CASE
	arg1, n_t0_uni, n_t0_gau, uniax, gauax = setInitialCondition()
	t_END = 8e-3
	t_INTERM = t_END/2.0
	tspan = array([0.0, t_INTERM, t_END])	
	xspan = arg1['x']
	mu0_t0 = integrate.trapz(n_t0_uni, xspan)
	mu1_t0 = integrate.trapz(n_t0_uni * xspan, xspan)

	y0 = n_t0_uni
	try:
		ySIM = pbeslv.caller_ode_MDF_growPure(y0, arg1, tspan)
	except:
		print('Error NDF growPure uniform')
		ySIM = zeros([tspan.shape[0], arg1['Npts']])
		
	k_t2 = where(tspan >= t_END)[0][0]
	plotmdfUni, = uniax.plot(xspan, ySIM[k_t2,:], 'gs:', lw = 2, label='$t_2=${} uniform'.format(t_END))
	mu0_t2 = integrate.trapz(ySIM[k_t2,:], xspan)
	mu1_t2 = integrate.trapz(ySIM[k_t2,:]*xspan, xspan)
	print('N(t_2) / N(t_0) = {}'.format(mu0_t2/mu0_t0))
	print('M(t_2) / M(t_0) = {}'.format(mu1_t2/mu1_t0))

	# GAUSS CASE
	y0 = n_t0_gau
	try:
		ySIM = pbeslv.caller_ode_MDF_growPure(y0, arg1, tspan)
	except:
		print('Error MDF growPure gaussian')
		ySIM = zeros([tspan.shape[0], arg1['Npts']])
		
	k_t2 = where(tspan >= t_END)[0][0]
	plotmdfGau, = gauax.plot(xspan, ySIM[k_t2,:], 'gs:', lw = 2, label='$t_2=${} gaussian'.format(t_END))
	mu0_t2 = integrate.trapz(ySIM[k_t2,:], xspan)
	mu1_t2 = integrate.trapz(ySIM[k_t2,:]*xspan, xspan)
	print('N(t_2) / N(t_0) = {}'.format(mu0_t2/mu0_t0))
	print('M(t_2) / M(t_0) = {}'.format(mu1_t2/mu1_t0))

	plt.savefig('mdf_growthOnly_PBE.svg')	


if __name__ == '__main__':
	main()