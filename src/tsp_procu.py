'''

'''

import numpy as np
import itertools as it

import mse_messaging as mse

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




## *  *  *  ##  *  *  *  ##

def xyDistMatrix(xs, ys):
    diffXs = np.stack([np.abs(x - xs) for x in xs])
    diffYs = np.stack([np.abs(y - ys) for y in ys])
    
    ## ##
    return np.sqrt(np.square(diffXs) + np.square(diffYs))  



def xyCountRoute(distanceMatrix):
    def underhood(route):
        runs = zip(route[:-1], route[1:])
        return sum([distanceMatrix[r] for r in runs] + [distanceMatrix[0, route[0]], distanceMatrix[route[-1], 0]]) 
    
    ## ##
    return underhood



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



##  *  *  *  WORKER's TARGET  *  *  *  ##

def cntWorkerFn(no, townIds, distMatrix, infDistance=999.0, name=None):
    def underhood(queIn, queOut):
        queOut.put(mse.InfoMessage(text="CNT{}: starting...".format(no)))
        
        minRoute = None
        minDistance = infDistance
        procCount = 0
        
        countDistanceFn = countHeadedRoutesFn(ids=townIds, distMatrix=distMatrix,
                                              infDistance=infDistance, name="Count_Distance_Fn_{}".format(no))
        
        running = True
        while running:
            msg = queIn.get()
            
            if isinstance(msg, mse.PoisonMessage):
                running = False
                
            elif isinstance(msg, mse.DataMessage):
                heads = msg.load   
                queOut.put(mse.InfoMessage(text="CNT{}: heads recieved: {}...".format(no, heads[0])))
            
                for head in heads:
                    route, distance, numCount = countDistanceFn(head)

                    if distance < minDistance:
                        minRoute = route
                        minDistance = distance
                        
                    procCount += numCount
                    queOut.put(mse.InfoMessage(text={"summary": "Routes Count Report",
                                                     "count": numCount}))
                        
                queOut.put(mse.DataMessage(data=(minRoute, minDistance, procCount)))
                queOut.put(mse.InfoMessage(text="CNT{}: candidate sent. Shutting down...".format(no)))
                queOut.put(mse.PoisonMessage())
                        
            else:
                queOut.put(mse.InfoMessage(text="CNT{}: recieved a message of unknown purpose".format(no)))
        
        ##
        return()
    
    ##
    if name is not None:
        underhood.__name__ = name
    return(underhood)










