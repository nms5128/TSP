'''


'''
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import multiprocessing as mp

##  *  *  *  MESSAGING  *  *  *  ##

class TspMsg:
    def __repr__(self):
        return("Abstract TSP message")
    def __str__(self):
        return("Abstract TSP message")



class PoisonMsg(TspMsg):
    def __repr__(self):
        return("TSP Poison message")
    def __str__(self):
        return("TSP Poison message")



class AskLoadMsg(TspMsg):
    def __repr__(self):
        return("TSP Ask for load message")
    def __str__(self):
        return("TSP Ask for load message")



class LoadMsg(TspMsg):
    def __init__(self, load):
        self.load = load
    def __repr__(self):
        return("TSP Load message: {}".format(self.load))
    def __str__(self):
        return("TSP Load message: {}".format(self.load))
        


class InfoMsg(TspMsg):
    def __init__(self, info):
        self.info = info
    def __repr__(self):
        return("TSP Info message: {}".format(self.info))
    def __str__(self):
        return("TSP Info message: {}".format(self.info))


'''
## TEST
genMsg = TspMsg()
poiMsg = PoisonMsg()
loaMsg = LoadMsg([42, 21])

print(isinstance(genMsg, TspMsg))
print(isinstance(genMsg, PoisonMsg))
print(isinstance(poiMsg, PoisonMsg))
print(isinstance(poiMsg, TspMsg))
print(isinstance(loaMsg, TspMsg))
print(loaMsg)
print(poiMsg)
'''

##  *  *  *  TOWNS CREATION  *  *  *  ##

def generateTowns(n=10, minval=-1.0, maxval=1.0):
    townNames = ["Bordertown", "Yharnam", "Eastheaven", "New Gettisburg", "Yahar'Gul (UV)", 
                 "Genty Town", "St. Gojiras", 
                 "(the FC of) Newark", "Wellington Wells",
                 "Sliabh Luachra", "Leithrim Fancy",
                 "Hemwick", "Battleground",
                 "Firelink Shrine", "Edge Knot City",
                 "San-Chelyabinsk", "Neuevasyuki", 
                 "Kryzhopl", "Bender's (Hold)", 
                 "Kuldakhar", "Redgrave"
                 ]
    
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
    for no, town in enumerate(towns):       
        ax.scatter(town[1][0], town[1][1])
        ax.text(x=town[1][0]+0.005, y=town[1][1]+0.005, s="{}. {}".format(no, town[0]))

    for town in towns:
        otherTowns = exclTown(towns, town)
        for t in otherTowns:
            ax.plot([town[1][0], t[1][0]], [town[1][1], t[1][1]], c="lightgrey", ls="--")
        
    if shortestTravel is not None:
        shortXs = [t[1][0] for t in shortestTravel]
        shortYs = [t[1][1] for t in shortestTravel]
        
        ax.plot(shortXs, shortYs, c="red")
        
    ax.grid(ls="--")



##  *  *  *  COUNT  *  *  *  ##

def xyDistMatrix(xs, ys):
    diffXs = np.stack([np.abs(x - xs) for x in xs])
    diffYs = np.stack([np.abs(y - ys) for y in ys])
    
    ##
    return(  np.sqrt(np.square(diffXs) + np.square(diffYs))  )



def xyCountRoute(distMatrix, name=None):
    def underhood(route):
        runs = zip(route[:-1], route[1:])
        return(  sum([distMatrix[run] for run in runs])  )
    
    ##
    if name is not None:
        underhood.__name__ = name
    return(underhood)



def headsFn(ids, headSize=1):
    ids = ids[1:]
    droppedNose = ids[-1]
    
    heads = list(it.permutations(ids, r=headSize))
    heads = [h for h in heads if h[0] != droppedNose]
    
    ##
    return(heads)



def rmTupleEnt(xs, x):
    if not isinstance(x, tuple):
        x = (x,)
        
    ##
    return(tuple(i for i in xs if i not in x))



def startEndShuffle(xs):
    return(list(it.chain.from_iterable(zip(xs, reversed(xs))))[:len(xs)])



def countHeadedRoutesFn(ids, distMatrix, infDistance=999.0, name=None):
    '''
    ids: List[Int], ordered id sequence, e.g. [0,1,2,3,4]
    '''
    def underhood(head):
        idRest = rmTupleEnt(ids[1:], head)
        
        gen = it.permutations(idRest)
        xyCountDistance = xyCountRoute(distMatrix)
        
        minRoute = None
        minDistance = infDistance
        numCount = 0
        
        running = True
        while running:
            try:
                routeVal = next(gen)
                route = (ids[0],) + head + routeVal + (ids[0],)
                
                if route[1] < route[-2]:
                    distance = xyCountDistance(route)

                    if distance < minDistance:
                        minRoute = route
                        minDistance = distance

                    numCount += 1
                
            except StopIteration:
                running = False       
        
        ##
        return(minRoute, minDistance, numCount)
    
    ##
    if name is not None:
        underhood.__name__ = name
    return(underhood)


'''
##  ##  TESTs
n = 11
headSize = 1

ids = list(range(n))
print(ids, end="\n\n")

## TEST 1: headsFn
if False:
    heads = headsFn(ids, headSize=headSize)
    print(heads, end="\n\n")
    
## TEST 2: rmTupleEnt
if False:
    heads = headsFn(ids, headSize=headSize)    
    head1 = heads[0]
    ret1 = rmTupleEnt(ids[1:], head1)
    print(head1, "==>", ret1)
    
    heads = headsFn(ids, headSize=2)    
    head2 = heads[-1]
    ret2 = rmTupleEnt(ids[1:], head2)
    print(head2, "==>", ret2)
    print()
    
## TEST 3: countHeadedRoutesFn
if False:
    townIds = list(range(11))
    headSize = 4
    xs = np.random.uniform(size=(len(townIds),), low=-1.0, high=1.0)
    ys = np.random.uniform(size=(len(townIds),), low=-1.0, high=1.0)
    distMatrix = tsp.xyDistMatrix(xs, ys)
    
    heads = headsFn(townIds, headSize=headSize)
    print("Number of {}-sized heads: {}".format(headSize, len(heads)))
    head4 = heads[4] 
    print("4-th head: {}".format(head4))
    idRest = rmTupleEnt(townIds[1:], head4)
    print("Combinations will be made with: {}".format(idRest))
    
    candidate = countHeadedRoutesFn(townIds, distMatrix)(head4)
    print(candidate)

## TEST 4: startEndShuffle
if False:
    for i in range(7,14):
        xs = list(range(i))
        xsShf = startEndShuffle(xs)
        print(xs, "==>", xsShf)

'''



##  *  *  *  WORKER's TARGET  *  *  *  ##

def cntWorkerFn(no, townIds, distMatrix, infDistance=999.0, name=None):
    def underhood(queIn, queOut):
        queOut.put(InfoMsg(info="CNT{}: starting...".format(no)))
        
        minRoute = None
        minDistance = infDistance
        procCount = 0
        
        countDistanceFn = countHeadedRoutesFn(ids=townIds, distMatrix=distMatrix,
                                              infDistance=infDistance, name="Count_Distance_Fn_{}".format(no))
        
        running = True
        while running:
            msg = queIn.get()
            
            if isinstance(msg, PoisonMsg):
                running = False
                
            elif isinstance(msg, LoadMsg):
                heads = msg.load   
                queOut.put(InfoMsg(info="CNT{}: heads recieved: {}...".format(no, heads[0])))
            
                for head in heads:
                    route, distance, numCount = countDistanceFn(head)

                    if distance < minDistance:
                        minRoute = route
                        minDistance = distance
                        
                    procCount += numCount
                    queOut.put(InfoMsg(info={"summary": "Routes Count Report",
                                             "count": numCount}))
                        
                queOut.put(LoadMsg(load=(minRoute, minDistance, procCount)))
                queOut.put(InfoMsg(info="CNT{}: candidate sent. Shutting down...".format(no)))
                queOut.put(PoisonMsg())
                        
            else:
                queOut.put(InfoMsg(info="CNT{}: recieved a message of unknown purpose".format(no)))
        
        ##
        return()
    
    ##
    if name is not None:
        underhood.__name__ = name
    return(underhood)


'''
no tests provided
'''



##  *  *  *  GENERAL PURPOSE  *  *  *  ##

def batchList(xs, batchSize=None, numBatches=None):
    '''
    only python used

    ARGUMENTS:
    xs: List[Any] - list of something
    batchSize: Integer
    numBatches: Integer

    RETURNS:
    List[List[Any]]
    '''
    count = len(xs)
    
    ##
    if (batchSize is not None) and (numBatches is None):
        numBatches = count // batchSize + int(count % batchSize != 0)
        
        slices = [slice(i*batchSize, (i+1)*batchSize, 1) for i in range(numBatches)]
        batches = [xs[slicei] for slicei in slices]
        
    ##
    elif (batchSize is None) and (numBatches is not None):
        minBatchSize = count // numBatches 
        
        adds = [1]*(count - minBatchSize*numBatches) + [0]*(numBatches - (count - minBatchSize*numBatches))
        batchSizes = [minBatchSize + a for a in adds]
        cumBatchSizes = [sum(batchSizes[:i]) for i in range(len(batchSizes)+1)]
        
        slices = [slice(start, stop, 1) for start, stop in zip(cumBatchSizes[:-1], cumBatchSizes[1:])]
        
        batches = [xs[slicei] for slicei in slices]
        batches = [b for b in batches if b]        
        
    ##
    else:
        raise Exception("Either 'batchSize' or 'numBatches' argument shall be defined")

    ##
    return(batches)



##  *  *  *  BRAIN DAMAGE FUNCTIONS  *  *  *  ##

def phuc(x):
    assert (x >= 1)

    def underhood(acc, x):
        if x == 1:
            return(acc)
        else:
            return(underhood(acc*x, x-1))

    ##
    return(underhood(1, x))



def flattenList(xs):
    if not isinstance(xs, list):
        xs = [xs]
        
    if all([not isinstance(x, list) for x in xs]):
        return(xs)
    else:
        acc = []
        for x in xs:
            if isinstance(x, list):
                acc += x
            else:
                acc += [x]
        return(flattenList(acc))