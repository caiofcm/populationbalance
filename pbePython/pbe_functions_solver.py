from numpy import *
from matplotlib import pyplot as plt
from scipy import integrate
import time


# AUXILIARIES FUNCTIONS TO IPYTHON NOTEBOOK FOR PBE PRESENTATION


## GROWTH ONLY PBE WITH MDF

def rhs_MDF_growPure(t, y, arg1):
    """ RHS PBE growth only based on Finite Differences,
    note: n = [n_0, n_1 , ... , n_npts]
    but because of the Boundary condition, n (and also rhs) does not include n_0
    y = n = [n_1, n_2, ..., n _npts] """
    rhs = empty_like(y)
    Npts = arg1['Npts']
    n = y[0:Npts - 1]
    x = arg1['x'][:]
    delta = arg1['delta']
    growth_fnc = arg1['growth_fnc']
    diff_growth_fnc = arg1['diff_growth_fnc']
    
    n0 = 0.0 #Boundary Condition
    dif_dndl1 = (n[2 - 1] - n0)/(2*delta)
    rhs[1 - 1] = -growth_fnc(x[1]) * dif_dndl1 - n[1 - 1]*diff_growth_fnc(x[1])
    for i in arange(2 - 1,Npts - 1):
        dif_dndl = 1.0/(2.0*delta)*(3*n[i] -4*n[i-1] + n[i-2]) - n[i]*diff_growth_fnc(x[i+1])
        rhs[i] = -growth_fnc(x[i+1]) * dif_dndl - n[i]*diff_growth_fnc(x[i+1])
    
    
    
    return rhs;
    
def caller_ode_MDF_growPure(y0, arg1, tspan):
    Npts = arg1['Npts']
    r = integrate.ode(rhs_MDF_growPure).set_integrator('vode', method='adams')
    r.set_initial_value(y0[1:], 0.0).set_f_params(arg1) #first point is not included (it is boundary cnd)
    #t_END = 8e-3
    #t_INTERM = t_END/2
   # tspan = array([0.0, t_INTERM, t_END])
    ySIM = empty([tspan.shape[0], Npts]) 
    #start_t = time.time()
    for k in arange(1,size(tspan)):
        ti = tspan[k]
        try:
            if (r.successful()):
                r.integrate(ti)
                ySIM[k,:] = hstack((0.0, transpose(r.y)))
                #print('t = ' + str(r.t))
            else:
                print(' Simulation error - Halted!' )
                break
        except:
            print('ODE sim error')
    return ySIM
    #print('ODE Solved in {} seconds'.format(time.time()-start_t))   
    
## GROWTH AND NUCLEATION PBE WITH MDF
def rhs_MDF_growNucl(t, y, arg1):
    """ RHS PBE growth and nucleation based on Finite Differences,
    note: n = [n_0, n_1 , ... , n_npts]
    but because of the Boundary condition, n (and also rhs) does not include n_0
    y = n = [n_1, n_2, ..., n _npts] """
    rhs = empty_like(y)
    Npts = arg1['Npts']
    n = y[0:Npts - 1]
    x = arg1['x'][:]
    delta = arg1['delta']
    growth_fnc = arg1['growth_fnc']
    diff_growth_fnc = arg1['diff_growth_fnc']
    nucl_fnc = arg1['nucl_fnc']
    
    n0 = nucl_fnc(x[0])/growth_fnc(x[0]) #Boundary Condition
    dif_dndl1 = (n[2 - 1] - n0)/(2*delta)
    rhs[1 - 1] = -growth_fnc(x[1]) * dif_dndl1 - n[1 - 1]*diff_growth_fnc(x[1])
    for i in arange(2 - 1,Npts - 1):
        dif_dndl = 1.0/(2.0*delta)*(3*n[i] -4*n[i-1] + n[i-2]) - n[i]*diff_growth_fnc(x[i+1])
        rhs[i] = -growth_fnc(x[i+1]) * dif_dndl - n[i]*diff_growth_fnc(x[i+1])
    
    return rhs;

def caller_ode_MDF_growNucl(y0, arg1, tspan):
    Npts = arg1['Npts']
    r = integrate.ode(rhs_MDF_growNucl).set_integrator('vode', method='adams')
    r.set_initial_value(y0[1:], 0.0).set_f_params(arg1) #first point is not included (it is boundary cnd)
    #t_END = 8e-3
    #t_INTERM = t_END/2
   # tspan = array([0.0, t_INTERM, t_END])
    ySIM = empty([tspan.shape[0], Npts]) 
    #start_t = time.time()
    for k in arange(1,size(tspan)):
        ti = tspan[k]
        if (r.successful()):
            r.integrate(ti)
            n0 = arg1['nucl_fnc'](arg1['x'][0]) / arg1['growth_fnc'](arg1['x'][0])
            ySIM[k,:] = hstack((n0, transpose(r.y)))
            #print('t = ' + str(r.t))
        else:
            raise(' Simulation error - Halted!' )
            break
    return ySIM
    
## MOVING SECTIONAL METHOD : GROWTH NUCLEATION AGGREGATION
def delta_kron(i,j):
    """Just a simple auxiliary function for computing kronecker delta"""
    if i == j:
        return 1
    else:
        return 0
    
def rhs_aggr_grow_nucl(t, y, arg1):
    """Function to give the rhs for the PBE equation with aggregation, nucleation
    and growth using the Moving Sectional Method"""
    
    rhs = zeros_like(y)
    Npts = arg1['Npts']
    x = y[0:Npts]
    N = y[Npts:]
    aggre_func = arg1['aggre_func']
    nucl_fnc = arg1['nucl_fnc']
    growth_fnc = arg1['growth_fnc']
    
    # First Term in Eq. 29: (i=Npts?)
    for i in arange(1, Npts): #does not apply for first bin this term
        if (isclose(x[i],x[i-1])):
            continue;
        if i == Npts - 1:
            xiplus1 = 1e20 #is that right?
        else:
            xiplus1 = x[i+1]
        term = 0.0
        for j in arange(0, i + 1):
            for k in arange(0, j + 1):
                t1 = 0.
                nu = x[j] + x[k]
                if (nu >= x[i-1]) and (nu <= xiplus1):
                    if (nu <= x[i]):
                        eta = (x[i-1] - nu)/(x[i-1]-x[i])
                    else: # (nu >= x[i]):
                        eta = (xiplus1 - nu)/(xiplus1-x[i])
                    aux1 = (1.0-1.0/2.0*delta_kron(j,k))*eta
                    qjk = aggre_func(x[j], x[k])
                    t1 = aux1*qjk*N[j]*N[k]
                    
                term += t1 #it is set to zero at k iteration
        rhs[Npts + i] = term
            

    # Second Term in Eq. 29:
    for i in arange(0, Npts):
        term = 0.0
        for k in arange(1, Npts):
            t2 = aggre_func(x[i],x[k])*N[k]
            term += t2
        rhs[Npts + i] += -N[i] * term
        
    #Third term is nucleation
    rhs[Npts + 0] += nucl_fnc()
    
    # The moving front term:
    rhs[0] = 0.5*(0.0 + growth_fnc(x[0])) #cntst growth
    for i in arange(1, Npts):
        rhs[i] = growth_fnc(x[i])

    return rhs;

def caller_ode_MovingSectional_AggreGrowNucl(y0, arg1, tspan, verbose=False):
    Npts = arg1['Npts']
    r = integrate.ode(rhs_aggr_grow_nucl).set_integrator('vode', method='adams')
    r.set_initial_value(y0, 0.0).set_f_params(arg1)
    d = {   'x' : [None]*(tspan.shape[0]), 'N' : [None]*(tspan.shape[0]), 
            'l': [None]*(tspan.shape[0]), 'n_num': [None]*(tspan.shape[0])}
    d['x'][0] = transpose(r.y[0:Npts])
    d['N'][0] = transpose(r.y[Npts:])
    d['l'][0] = arg1['lspan']
    d['n_num'][0] = d['N'][0] / (d['l'][0][1:] - d['l'][0][0:-1])
    start_t = time.time()
    Npivot = Npts
    lprevious = copy(arg1['lspan'])
    for k in arange(1,size(tspan)):
        ti = tspan[k]
        lt = empty_like(lprevious)
        
        if (r.successful()):
            r.integrate(ti)
            xt = r.y[0:Npivot]
            Nt = r.y[Npivot:]
            delta_x = xt[-1] - d['x'][k-1][-1] #constant growth
            
            #CNSTANT GROWTH lt out of rhs and manually integrated:
            lt[0] = lprevious[0] #dl0dt = 0
            lt[1:] = lprevious[1:] + delta_x #regular growth 
            
            nNum_t = Nt / (lt[1:] - lt[0:-1])

            d['x'][k] = transpose(xt)
            d['N'][k] = transpose(Nt)
            d['l'][k] = transpose(lt)
            d['n_num'][k] = transpose(nNum_t)
             
            if verbose : print('k={} and t = {}'.format(k,r.t)) 
            
            # ADD A BIN
            Npivot+=1
            arg1['Npts'] = Npivot
            lprevious = hstack((0.0, lt))
            xnow = (lprevious[1:] + lprevious[0:-1])/2.0
            Nnow =  hstack((0.0, Nt))
            y0now = hstack((xnow, Nnow))
            r.set_initial_value(y0now, ti).set_f_params(arg1)
            
        else:
            print(' Simulation error - Halted!' )
            break
    if verbose : print('ODE Solved in {} seconds'.format(time.time()-start_t))
    
    return d


