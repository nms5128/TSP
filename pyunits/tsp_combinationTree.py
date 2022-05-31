'''

'''

import itertools as it
import tsp.liball as tsp

class CombinationsTree:
    def __init__(self, foundation, root, branches):
        self.foundation = foundation
        self.root = root

        if any([b > foundation for b in branches]):            
            self.branches = branches        
            self.trees = [CombinationsTree(foundation=foundation, root=r, branches=tsp.removeListItem(branches, r)) \
                          for r in branches
                          if any([b > foundation for b in tsp.removeListItem(branches, r)]) \
                             or len(branches) <= 1]
        else:
            self.branches = []
            self.trees = []
            
    def __str__(self):
        return("Tree: Root: {}, Branches {}, Found: {}".format(self.root, self.branches, self.foundation))
        
    def __repr__(self):
        return("Tree: Root: {}, Branches {}, Found: {}".format(self.root, self.branches, self.foundation))
    
    def extractRoutes(self):
        if self.trees == []:
            if self.foundation == self.root:
                return([])
            else:
                return([[self.root]])
        else:
            return(tsp.prependBeginToEnds(begin=[self.root],
                                          ends=list(it.chain.from_iterable([t.extractRoutes() for t in self.trees]))))
        
        
class MainCombinationsTree:
    def __init__(self, xs):
        self.trees = [CombinationsTree(x, x, tsp.removeListItem(xs, x)) for x in xs]
        
    def extractRoutes(self):
        routes = list(it.chain.from_iterable([t.extractRoutes() for t in self.trees]))
        return(routes)
        


''''
## TEST
import time

n = 11

timeStart = time.time()
notOptimized = list(it.permutations(iterable=range(1, n+1), r=n))
t1 = time.time() - timeStart

timeStart = time.time()
T = MainCombinationsTree(range(1, n+1))
optimized = T.extractRoutes()
t2 = time.time() - timeStart

print("{} Vs. {}".format(len(optimized), len(notOptimized)))
print("Time Not-Optimzed {} Vs. Time Optimized: {} sec.".format("%.2f"%t1, "%.2f"%t2))
'''