import sys
import csv
import numpy
import getopt
import math

def usage():
    print '$> python genDNA.py <required args> [optional args]\n' + \
        '\t-c <#>\t\tNumber of clusters to generate\n' + \
        '\t-p <#>\t\tNumber of points per cluster\n' + \
        '\t-o <file>\tFilename for the output of the raw data\n' + \
        '\t-v [#]\t\tMaximum coordinate value for points\n'  


def euclideanDistance(p1, p2):
    '''
    Takes two 2-D points and computes the Euclidean distance between them.
    '''
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + \
                    math.pow((p2[1] - p1[1]), 2))

def tooClose(point, points, minDist):
    '''
    Computes the euclidean distance between the point and all points
    in the list, and if any points in the list are closer than minDist,
    this method returns true.
    '''
    for pair in points:
        if euclideanDistance(point, pair) < minDist:
                return True

    return False

def handleArgs(args):
    # set up return values
    numClusters = -1
    numPoints = -1
    output = None
    maxValue = 10

    try:
        optlist, args = getopt.getopt(args[1:], 'c:p:v:o:')
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    for key, val in optlist:
        # first, the required arguments
        if   key == '-c':
            numClusters = int(val)
        elif key == '-p':
            numPoints = int(val)
        elif key == '-o':
            output = val
        # now, the optional argument
        elif key == '-v':
            maxValue = float(val)

    # check required arguments were inputted  
    if numClusters < 0 or numPoints < 0 or \
            maxValue < 1 or \
            output is None:
        usage()
        sys.exit()
    return (numClusters, numPoints, output, \
            maxValue)

def drawOrigin(maxValue):
	# A,C,G,T encoded as 1-4
    return numpy.random.random_integers(1, 4, maxValue)

# start by reading the command line
numClusters, \
numPoints, \
output, \
maxValue = handleArgs(sys.argv)

fout = open(output.strip(), "w")
writer = csv.writer(fout)

# step 1: generate each 2D centroid
centroids_radii = []
minDistance = 0
for i in range(0, numClusters):
    centroid_radius = drawOrigin(maxValue)
    # is it far enough from the others?
    while (tooClose(centroid_radius, centroids_radii, minDistance)):
        centroid_radius = drawOrigin(maxValue)
    centroids_radii.append(centroid_radius)

# step 2: generate the points for each centroid
points = []
minClusterVar = 0.5
maxClusterVar = 1.5
for i in range(0, numClusters):
    # compute the variance for this cluster
    variance = numpy.random.uniform(minClusterVar, maxClusterVar)
    cluster = centroids_radii[i]
    for j in range(0, numPoints):
        # generate a 2D point with specified variance
        # point is normally-distributed around centroids[i]
		vec = numpy.random.normal(cluster, variance, maxValue)
		vec = numpy.around(vec)
		vec[vec < 1] = 1
		vec[vec > 4] = 4
		vec = vec.astype(int)
		# write the points out
		writer.writerow(vec)
fout.close()
