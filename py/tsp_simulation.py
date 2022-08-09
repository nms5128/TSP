'''

'''

import numpy as np
import matplotlib.pyplot as plt

def generateTowns(n=10, minval=-1.0, maxval=1.0):
    townNames = ["Bordertown", "Columbia", "Rapture", "Yharnam", "Yahar'Gul (UV)", 
                 "Genty Town", "Funkytown", "St. Gojiras",
                 "(the FC of) Newark", "Nite-City", 
                 "Starlink Shrine", "Edge Knot City", "Mountain Knot City", "Vault 13", "Hemwick",
                 "San-Chelyabinsk", "Neuevasyuki", "Houston (WGP)", 
                 "Kryzhopl", "Bender's (Hold)", "Battleground"
                 "Zion", 
                 "Kuldakhar", "Eastheaven", "New Gettisburg", "Wellington Wells", "Redgrave",
                 "Sliabh Luachra", "Leithrim Fancy"]
    
    points = np.random.uniform(size=(n,2), low=minval, high=maxval)
    points = [tuple(i) for i in points]
    
    towns = [(name, coords) for name,coords in zip(townNames[:n], points)]
    return(towns)



def exclTown(towns, town):
    if not isinstance(town, list):
        town = [town]
    ## 
    return([t for t in towns if t not in town])



def plotTowns(towns, shortestTravel=None, plotSize=10):
    '''
    towns: List[Tuple(no, name, Tuple(coordX, coordY))] - full unsorted list of towns
    shortestTravel: Tuple(  Tuple(no, name, Tuple(coordX, coordY))), int  ) - one of shortest travels found and its length
    '''
    fig, ax = plt.subplots(1,1, figsize=(plotSize, plotSize))
    for town in towns:
        otherTowns = exclTown(towns, town)
        for t in otherTowns:
            ax.plot([town[2][0], t[2][0]], [town[2][1], t[2][1]], c="lightgrey")

    for town in towns:       
        ax.scatter(town[2][0], town[2][1])
        ax.text(x=town[2][0]+0.005, y=town[2][1]+0.005, s="{}: {}".format(town[0],town[1]))
        
    if shortestTravel is not None:
        shortXs = [t[2][0] for t in shortestTravel]
        shortYs = [t[2][1] for t in shortestTravel]
        
        ax.plot(shortXs, shortYs, c="red")
        
    ax.grid(ls="--")
