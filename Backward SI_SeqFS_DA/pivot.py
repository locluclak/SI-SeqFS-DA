import numpy as np
from gendata import generate
import OptimalTransport
import BackwardSelection as FS
import overconditioning 
import parametric
from scipy.linalg import block_diag
from mpmath import mp
mp.dps = 500

import time
def compute_p_value(intervals, etaT_Y, etaT_Sigma_eta):
    denominator = 0
    numerator = 0

    for i in intervals:
        leftside, rightside = i
        if leftside <= etaT_Y <= rightside:
            numerator = denominator + mp.ncdf(etaT_Y / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
        denominator += mp.ncdf(rightside / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))

    cdf = float(numerator / denominator)
    # print(cdf)
    # compute two-sided selective p_value
    return 2 * min(cdf, 1 - cdf)

def pvalue_SI(ns, nt, p, true_betaS, true_betaT, k):
    """Return final p_value"""
    # np.random.seed(seed)

    # Generate data
    Xs, Xt, Ys, Yt, Sigma_s, Sigma_t = generate(ns, nt, p, true_betaS, true_betaT)

    #Concatenate data (X, Y)
    Xs_ = np.concatenate((Xs, Ys), axis = 1)
    Xt_ = np.concatenate((Xt, Yt), axis = 1)

    #Concatenate data into a bunch (XsYs, XtYt).T
    XsXt_ = np.concatenate((Xs_, Xt_), axis= 0)

    #Bunch of Xs and Xt
    X = np.concatenate((Xs, Xt), axis= 0)
    # Bunch of Ys & Yt
    Y = np.concatenate((Ys, Yt), axis= 0)

    Sigma = block_diag(Sigma_s, Sigma_t)

    h = np.concatenate((np.ones((ns, 1))/ns, np.ones((nt,1))/nt), axis = 0) 
    S = OptimalTransport.convert(ns,nt)
    # remove last row
    S_ = S[:-1].copy()
    h_ = h[:-1].copy()
    # Gamma drives source data to target data 
    GAMMA, basis_var = OptimalTransport.solveOT(ns, nt, S_, h_, XsXt_).values()

    # Bunch of Xs Xt after transforming
    Xtilde = np.dot(GAMMA, X)
    Ytilde = np.dot(GAMMA, Y)

    Sigmatilde = GAMMA.T.dot(Sigma.dot(GAMMA))
    # Best model from 1...p models by AIC criterion
    if k == 'AIC':
        SELECTION_F = FS.SelectionAICforBS(Ytilde, Xtilde, Sigmatilde)
    elif k == 'BIC':
        SELECTION_F = FS.SelectionBIC(Ytilde, Xtilde, Sigmatilde)
    elif k == 'Adjusted R2':
        SELECTION_F = FS.SelectionAdjR2(Ytilde, Xtilde)
    else:
        SELECTION_F = FS.fixedBS(Ytilde, Xtilde, k)[0]
    Xt_M = Xt[:, sorted(SELECTION_F)].copy()
    print('Selected feature jth:',SELECTION_F, 'True beta:', true_betaT[sorted(SELECTION_F)].reshape(1,-1))
    # Compute eta
    # jtest = np.random.choice(range(len(SELECTION_F)))
    ls_pvalue = []
    for jtest in range(len(SELECTION_F)):
        e = np.zeros((len(SELECTION_F), 1))
        e[jtest][0] = 1

        # Zeta cut off source data in Y
        Zeta = np.concatenate((np.zeros((nt, ns)), np.identity(nt)), axis = 1)
        
        # eta constructed on Target data
        eta = np.dot(e.T , np.dot(np.dot(np.linalg.inv(np.dot(Xt_M.T, Xt_M)), Xt_M.T), Zeta)) 
        eta = eta.reshape((-1,1))
        etaT_Sigma_eta = np.dot(np.dot(eta.T , Sigma) , eta).item()
        
        # Change y = a + bz
        I_nplusm = np.identity(ns+nt)
        b = np.dot(Sigma, eta) / etaT_Sigma_eta
        a = np.dot((I_nplusm - np.dot(b, eta.T)), Y)

        # Test statistic
        etaTY = np.dot(eta.T, Y).item()
        # print(f"etay: {etaTY}")
        if type(k) == int:
            finalinterval = parametric.para_DA_BS(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F)
        else:
            finalinterval = parametric.para_DA_BSwith_stoppingCriteria(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F,typeCrit=k)
        # print(finalinterval)
        selective_p_value = compute_p_value(finalinterval, etaTY, etaT_Sigma_eta)
        ls_pvalue.append(selective_p_value)
    return ls_pvalue

def pvalue_SI_randcheck(ns, nt, p, true_betaS, true_betaT, k):
    """Return final p_value"""
    # np.random.seed(seed)

    # Generate data
    Xs, Xt, Ys, Yt, Sigma_s, Sigma_t = generate(ns, nt, p, true_betaS, true_betaT)

    #Concatenate data (X, Y)
    Xs_ = np.concatenate((Xs, Ys), axis = 1)
    Xt_ = np.concatenate((Xt, Yt), axis = 1)

    #Concatenate data into a bunch (XsYs, XtYt).T
    XsXt_ = np.concatenate((Xs_, Xt_), axis= 0)

    #Bunch of Xs and Xt
    X = np.concatenate((Xs, Xt), axis= 0)
    # Bunch of Ys & Yt
    Y = np.concatenate((Ys, Yt), axis= 0)

    Sigma = block_diag(Sigma_s, Sigma_t)

    h = np.concatenate((np.ones((ns, 1))/ns, np.ones((nt,1))/nt), axis = 0) 
    S = OptimalTransport.convert(ns,nt)
    # remove last row
    S_ = S[:-1].copy()
    h_ = h[:-1].copy()
    # Gamma drives source data to target data 
    GAMMA, basis_var = OptimalTransport.solveOT(ns, nt, S_, h_, XsXt_).values()

    # Bunch of Xs Xt after transforming
    Xtilde = np.dot(GAMMA, X)
    Ytilde = np.dot(GAMMA, Y)

    Sigmatilde = GAMMA.T.dot(Sigma.dot(GAMMA))
    # Best model from 1...p models by AIC criterion
    if k == 'AIC':
        SELECTION_F = FS.SelectionAICforBS(Ytilde, Xtilde, Sigmatilde)
    elif k == 'BIC':
        SELECTION_F = FS.SelectionBIC(Ytilde, Xtilde, Sigmatilde)
    elif k == 'Adjusted R2':
        SELECTION_F = FS.SelectionAdjR2(Ytilde, Xtilde)
    else:
        SELECTION_F = FS.fixedBS(Ytilde, Xtilde, k)[0]
    Xt_M = Xt[:, sorted(SELECTION_F)].copy()
    # print('Selected feature jth:',SELECTION_F, 'True beta:', true_betaT[sorted(SELECTION_F)].reshape(1,-1))
    # Compute eta
    jtest = np.random.choice(range(len(SELECTION_F)))
    # ls_pvalue = []
    # for jtest in range(len(SELECTION_F)):
    e = np.zeros((len(SELECTION_F), 1))
    e[jtest][0] = 1

    # Zeta cut off source data in Y
    Zeta = np.concatenate((np.zeros((nt, ns)), np.identity(nt)), axis = 1)
    
    # eta constructed on Target data
    eta = np.dot(e.T , np.dot(np.dot(np.linalg.inv(np.dot(Xt_M.T, Xt_M)), Xt_M.T), Zeta)) 
    eta = eta.reshape((-1,1))
    etaT_Sigma_eta = np.dot(np.dot(eta.T , Sigma) , eta).item()
    
    # Change y = a + bz
    I_nplusm = np.identity(ns+nt)
    b = np.dot(Sigma, eta) / etaT_Sigma_eta
    a = np.dot((I_nplusm - np.dot(b, eta.T)), Y)

    # Test statistic
    etaTY = np.dot(eta.T, Y).item()
    # print(f"etay: {etaTY}")
    if type(k) == int:
        finalinterval = parametric.para_DA_BS(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F)
    else:
        finalinterval = parametric.para_DA_BSwith_stoppingCriteria(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F,typeCrit=k)
    # print(finalinterval)
    selective_p_value = compute_p_value(finalinterval, etaTY, etaT_Sigma_eta)
        # ls_pvalue.append(selective_p_value)
    return selective_p_value
