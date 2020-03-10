import numpy as np

def catchment(flowDir, initPt):
    # dictionary to lookup the neighbors based on flow direction array
    dir_map = {
        1: {
            'dir': 5, # flattened index of the value to reference when searching for a grid
            'idx': np.unravel_index(5,[3,3]) # index in reference to the center pixel to adjust for
        },
        2: {
            'dir': 8,
            'idx': np.unravel_index(8,[3,3])
        },
        3: {
            'dir': 7,
            'idx': np.unravel_index(7,[3,3])
        },
        4: {
            'dir': 6,
            'idx': np.unravel_index(6,[3,3])
        },
        5: {
            'dir': 4,
            'idx': np.unravel_index(3,[3,3])
        },
        6: {
            'dir': 1,
            'idx': np.unravel_index(0,[3,3])
        },
        7: {
            'dir': 2,
            'idx': np.unravel_index(1,[3,3])
        },
        8: {
            'dir': 3,
            'idx': np.unravel_index(2,[3,3])
        },
    }


    catchIdx = [] # list of pixels that contribute to starting point
    pts = [initPt] # list of points to search which neighboring cells flow through

    searching = True
    while searching:
        nextIdxes = [] # blank list to append for next iter step
        # loop through all of the points and find which neighbors contribute
        for p in pts:
            flows = []
            # get neighbors
            for k,v in dir_map.items():
                flows.append(flowDir[p[0]-1:p[0]+2,p[1]-1:p[1]+2,k].ravel()[v['dir']])

            # find the direction of neighbors and append
            for j in np.where(np.array(flows)>0)[0]:
                dy,dx = [d-1 for d in dir_map[j+1]['idx']]
                ni = [p[0]+dy,p[1]+dx]
                nextIdxes.append(ni)

        # append the contributing cells to the total list
        catchIdx.extend(nextIdxes)

        # reset the pts variable for next iter
        pts = nextIdxes

        if len(nextIdxes) <= 0 :
            searching = False

    # transform the catchment indexes to 2-d array
    catchOut = np.zeros_like(flowDir[:,:,-1])
    catchOut[tuple(zip(*catchIdx))] = 1

    return catchOut
