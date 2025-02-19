import overconditioning
import numpy as np
import OptimalTransport
import ForwardSelection
import intersection

def para_DA_FSwithStoppingCriterion(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F, k, seed = 0):
    TD = []
    detectedinter = []
    z =  -20
    zmax = 20
    countitv=0
    while z < zmax:
        z += 0.0001

        for i in range(len(detectedinter)):
            if detectedinter[i][0] <= z <= detectedinter[i][1]:
                z = detectedinter[i][1] + 0.0001
                detectedinter = detectedinter[i:]
                break
        if z > zmax:
            break
        # print(z)
        Ydeltaz = a + b*z

        XsXt_deltaz = np.concatenate((X, Ydeltaz), axis= 1).copy()
        GAMMAdeltaz, basis_var_deltaz = OptimalTransport.solveOT(ns, nt, S_, h_, XsXt_deltaz).values()

        Xtildeinloop = np.dot(GAMMAdeltaz, X)
        Ytildeinloop = np.dot(GAMMAdeltaz, Ydeltaz)

        Sigmatilde_deltaz = GAMMAdeltaz.T.dot(Sigma.dot(GAMMAdeltaz))
        if k == 'AIC':
            SELECTIONinloop = ForwardSelection.SelectionAIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
        elif k == 'BIC':
            SELECTIONinloop = ForwardSelection.SelectionBIC(Ytildeinloop, Xtildeinloop, Sigmatilde_deltaz)
        elif k == 'Adjusted R2':
            SELECTIONinloop = ForwardSelection.SelectionAdjR2(Ytildeinloop, Xtildeinloop)
        
        intervalinloop = overconditioning.OC_Crit_interval(ns, nt, a, b, XsXt_deltaz, 
                                                            Xtildeinloop, Ytildeinloop, Sigmatilde_deltaz, 
                                                            basis_var_deltaz, S_, h_, 
                                                            SELECTIONinloop, GAMMAdeltaz, k)
        countitv += 1
        # print(f"intervalinloop: {intervalinloop}")
        detectedinter = intersection.Union(detectedinter, intervalinloop)

        if sorted(SELECTIONinloop) != sorted(SELECTION_F):

            continue

        TD = intersection.Union(TD, intervalinloop)
    return TD


def para_DA_FSwithfixedK(ns, nt, a, b, X, Sigma, S_, h_, SELECTION_F):
    TD = []
    detectedinter = []
    z =  -20
    zmax = 20
    countitv = 0
    while z < zmax:
        z += 0.0001

        for i in range(len(detectedinter)):
            if detectedinter[i][0] <= z <= detectedinter[i][1]:
                z = detectedinter[i][1] + 0.0001
                detectedinter = detectedinter[i:]
                break
        if z > zmax:
            break
        # print(z)
        Ydeltaz = a + b*z

        XsXt_deltaz = np.concatenate((X, Ydeltaz), axis= 1).copy()
        GAMMAdeltaz, basis_var_deltaz = OptimalTransport.solveOT(ns, nt, S_, h_, XsXt_deltaz).values()

        Xtildeinloop = np.dot(GAMMAdeltaz, X)
        Ytildeinloop = np.dot(GAMMAdeltaz, Ydeltaz)

        Sigmatilde_deltaz = GAMMAdeltaz.T.dot(Sigma.dot(GAMMAdeltaz))
        SELECTIONinloop = ForwardSelection.fixedSelection(Ytildeinloop, Xtildeinloop, len(SELECTION_F))[0]

        lst_SELECk_deltaz, lst_P_deltaz = ForwardSelection.list_residualvec(Xtildeinloop, Ytildeinloop)


        
        intervalinloop, itvda, itvfs = overconditioning.OC_fixedFS_interval(ns, nt, a, b, XsXt_deltaz, 
                                                            Xtildeinloop, Ytildeinloop, Sigmatilde_deltaz, 
                                                            basis_var_deltaz, S_, h_, 
                                                            SELECTIONinloop, GAMMAdeltaz)

        countitv +=1
        detectedinter = intersection.Union(detectedinter, intervalinloop)

        if sorted(SELECTIONinloop) != sorted(SELECTION_F):
            # print(f"M != Mz | {SELECTIONinloop} | fs: {itvfs} | da: {itvda}")
            continue

        # print(SELECTIONinloop)
        # print(f"Matched - fs: {itvfs} - da: {itvda}")
        TD = intersection.Union(TD, intervalinloop)
    
    return TD

