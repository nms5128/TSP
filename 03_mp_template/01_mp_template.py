"""
A fancy description to be added
"""

def parseArgs():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_towns",
                        type=int,
                        default=7,
                        help="number of cities for TSP solution")
    opt = parser.parse_args()
    return opt



def main(opt):
    import os
    import sys
    import time

    import itertools as it
    import numpy as np

    codeLocation = os.path.join(os.getcwd(), "..", "src")
    sys.path.append(codeLocation)

    import tsp_sim_randomTowns as twn
    import tsp_procu as tsp


    n = opt.num_towns
    towns = twn.generateTowns(n=n)
    xs = [t[1][0] for t in towns]
    ys = [t[1][1] for t in towns]
    distmat = tsp.xyDistMatrix(xs, ys)

    ## ##
    routes = it.permutations(range(1, n))

    distanceMin = np.finfo(np.float32).max
    routeMin = None
    counter = 0

    timeStart = time.time()
    running = True
    while running:
        try:
            route = next(routes)
            if route[0] < route[-1]:
                distance = tsp.xyCountRoute(distmat)(route)

                if distance < distanceMin:
                    distanceMin = distance
                    routeMin = route

                counter += 1
                print("Progress: {}/Unknown".format(counter), end="\r")
            
        except Exception as e:
            print(e)
            running = False
            
    timeTaken = time.time() - timeStart
    print("Ok! Time taken: {} sec.".format("%.2f"%timeTaken))

    print("Shortest route: {}".format(routeMin))
    print("Shortest distance: {}".format(distanceMin))


if __name__ == "__main__":
    opt = parseArgs()
    main(opt)