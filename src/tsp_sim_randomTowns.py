'''

'''

import numpy as np
import itertools as it
import matplotlib.pyplot as plt

import tsp_procu as tsp


##  *  *  *  TOWNS CREATION  *  *  *  ##

def generateTowns(n=10, minval=-1.0, maxval=1.0):
    townNames = ["Bordertown", "Yharnam", "Eastheaven", "New Gettisburg", "Yahar'Gul (UV)", 
                 "Genty Town", "St. Gojiras", 
                 "(the FC of) Newark", "Wellington Wells",
                 "Sliabh Luachra", "Leithrim Fancy",
                 "Hemwick", "Battleground", "New Midgard",
                 "Firelink Shrine", "Edge Knot City", "Mountain K.N. (Knot City)"
                 "San-Chelyabinsk", "Neuevasyuki", 
                 "Kryzhopl", "Bender's (Hold)", 
                 "Kuldakhar", "Redgrave"
                 ]

    if n > len(townNames):
        nm = n // len(townNames) + 1
        townNames = list(it.chain.from_iterable([["{}_{}".format(townName, m) for townName in townNames] for m in range(nm)]))
    
    ## Generate coords randomly
    points = np.random.uniform(size=(n,2), low=minval, high=maxval)
    points = [tuple(i) for i in points]
    
    ## Return
    towns = [(no, name, coords) for no, name, coords in zip(range(n), townNames[:n], points)]
    return towns


def exclTown(towns, town):
    if not isinstance(town, list):
        town = [town]

    ## 
    return [t for t in towns if t not in town]



##  *  *  *  OUTPUT  *  *  *  ##

def plotTowns(towns, ax=None, plotSize=10, nk=4, shortestRoute=None):
    '''
    
    '''
    nk = min(nk, len(towns))
    
    distmat = tsp.xyDistMatrix([town[2][0] for town in towns],
                               [town[2][1] for town in towns])

    knearest = {i: np.argsort(distmat[i])[1:nk+1] \
                for i in range(distmat.shape[-1])}

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(plotSize, plotSize))

    for no, town in enumerate(towns):       
        ax.scatter(town[2][0], town[2][1])
        ax.text(x=town[2][0]+0.005, y=town[2][1]+0.005, s="{}. {}".format(town[0], town[1]))

        for ko in knearest[no]:
            ax.plot([town[2][0], towns[ko][2][0]], 
                    [town[2][1], towns[ko][2][1]], 
                    c="lightgrey", ls="-")
            
    if shortestRoute is not None:
        shortXs = [t[2][0] for t in shortestRoute]
        shortYs = [t[2][1] for t in shortestRoute]
        
        ax.plot(shortXs, shortYs, c="red")
        
    ax.axis("equal")
    ax.grid(ls="--")








