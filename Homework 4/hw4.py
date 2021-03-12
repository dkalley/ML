import math
import random
import time
import matplotlib.pyplot as plt
import pandas as pd

############################################################
#   Distance Methods
############################################################
def euclidean(point,centroid):
    p = [point[i] for i in range(len(point)-1)]
    return math.dist(p,centroid)

# difference in x1 + difference in x2 + ... + difference in xn
def manhattan(point,centroid):
    total = 0
    for i in range(len(point)-1):
        total = total + abs(point[i] - centroid[i])
    return total

def cosine_coefficient(point,centroid):
    xy = 0
    x2 = 0
    y2 = 0
    for i in range(len(point)-1):
        xy = xy + point[i]*centroid[i]
        x2 = x2 + point[i]*point[i]
        y2 = y2 + centroid[i]*centroid[i]
    if x2 == 0 or y2 == 0:
        return 1
    return (xy / (math.sqrt(x2)*math.sqrt(y2)))

def cosine(point,centroid):
    return 1 - cosine_coefficient(point,centroid)

def jaccard_coefficient(point,centroid):
    min_sum = 0
    max_sum = 0
    for i in range(len(point)-1):
        min_sum = min_sum + min(point[i],centroid[i])
        max_sum = max_sum + max(point[i],centroid[i])
    return min_sum/max_sum

def jaccard(point,centroid):
    return (1 - jaccard_coefficient(point,centroid))

############################################################
#   Printing Methods
############################################################
def initial_plot(plt,centroids,points, pointLabels):    
    labels = []
    # Plot points and labels
    for i in range(len(points)):
        plt.plot(points[i][0],points[i][1],'bo')
        if pointLabels:
            p = (points[i][0],points[i][1])
            plt.annotate(points[i][-1], p,textcoords='offset points', xytext=(0,10),ha='center')
    
    # Plot initial centroids
    colors = ['rx','gx','cx','mx']
    for i,centroid in enumerate(centroids):
        plt.plot(centroid[0],centroid[1],colors[i],markersize=20,linewidth=5)

    plt.show()

def determine_radius(origin, cluster):
    # Find the distance of the point in cluster farthest from origin
    distance = 0
    for point in cluster:
        new_distance = euclidean(point, origin)
        if new_distance > distance:
            distance = new_distance

    return distance

def plot_clusters(plt, centroids, clusters, clusternames, pointLabels):
    # Plot new centroids
    colorsc = ['rx','gx','cx','mx']
    for i,centroid in enumerate(centroids):
        plt.plot(centroid[0],centroid[1],colorsc[i],markersize=20,linewidth=5)
    
    # Plot each point in the correct cluster
    colorsp = ['ro','go','co','mo']
    for i in range(len(clusters)):
        for p in clusters[i]:
            plt.plot(p[0],p[1],colorsp[i])
            if pointLabels:
                point = (p[0],p[1])
                plt.annotate(p[-1],point,textcoords='offset points', xytext=(0,10),ha='center')

    # Plot circles around clusters, use centroids as center points
    ax = plt.gca()

    colors = ['r','g','c','m']
    for i,centroid in enumerate(centroids):
        radius = determine_radius(centroid,clusters[i])
        circle = plt.Circle((centroid[0],centroid[1]),radius,color=colors[i],fill=False)
        ax.add_patch(circle)
        point = (centroid[0],centroid[1])
        if clusternames:
            ax.annotate(clusternames[i],point, textcoords='offset points', xytext=(0,radius*15),ha='center',size=15)

    plt.show()

############################################################
#   Stopping Criteria Methods
############################################################
def centroid_based(centroids,new_centroids,iteration,max_iter,sse):
    return (centroids != new_centroids)

def iteration_based(centroids,new_centroids,iteration,max_iter,sse):
    return iteration < max_iter

def sse_based(centroids,new_centroids,iteration,max_iter,sse):
    return (sse[1] < sse[0]) and (iteration < max_iter)

############################################################
#   Helper Methods
############################################################
def cluster_info(cluster, i):
    if len(cluster) == 0:
        return 'Empty', 0, 0, 0

    # Create a dictionary of counts
    counts = {}

    # Count labels
    for point in cluster:
        if point[-1] not in counts:
            counts[point[-1]] = 1
        else:
            counts[point[-1]] = counts[point[-1]] + 1

    name = max(counts, key=counts.get)
    correct = counts[name]
    incorrect = len(cluster) - correct
    accuracy = correct / (incorrect + correct)

    return name, accuracy, correct, incorrect
        
def assign(dataset, centroids, clusters, f):
    # For each point in the dataset
    for point in dataset:
        # Determine distance to all centroids
        dist = [0 for i in range(len(centroids))]
        for i, centroid in enumerate(centroids):
            dist[i] = f(point, centroid)
        
        # Determine which centroid is closest
        idx = 0
        for i in range(1,len(dist)):
            if dist[i] < dist[idx]:
                idx = i

        # Add point to that cluster
        clusters[idx].append(point)
    return clusters

def recalculate_cluster(clusters,num_attributes):
    new_centroids = []

    # Recalculate centroids
    for cluster in clusters:
        sums = [0 for i in range(num_attributes)]
        for p in cluster:
            for i in range(len(p)-1):
                sums[i] = sums[i] + p[i]

        if len(cluster) == 0:
            new_centroids.append([0 for i in range(len(sums))])
        else:
            new_centroids.append([sums[i]/len(cluster) for i in range(len(sums))])

    return new_centroids

def calculate_sse(centroids,clusters):
    sse_value = 0
    for i,cluster in enumerate(clusters):
        for point in cluster:
            p = [point[i] for i in range(len(point)-1)]
            sse_value = sse_value + math.dist(p,centroids[i])*math.dist(p,centroids[i])

    return sse_value

############################################################
#   kmeans Methods
############################################################
# Default Arguments
#   distance: Euclidean
#   initCentroids: None
#   printIterations: None
#   Stopping Criteria: Centroid Based
#   Max Iterations: 100
#   Plot: True
#   Name Clusters: False
#   Print Point Labels: True
#   Time execution: False (Should not be used if printing iterations)
def kmeans(dataset,numClusters,distanceFunction=euclidean,initCentroids=None,printIteration=[],stoppingCriteria=centroid_based,max_iter=100,plot=True,clusterName=False,pointLabels=True,timer=False):
    # Either initialize centroids with a random sample or used pased centroids
    if initCentroids == None or len(initCentroids) < numClusters:
        # randomly select k initial centroids
        random.seed(time.time())
        data = []
        for i in range(len(dataset)):
            line = []
            for j in range(len(dataset[0])-1):
                line.append(dataset[i][j])
            data.append(line)
        centroids = random.sample(data, numClusters)
    else:
        centroids = initCentroids

    # Plot the points and initial centroids
    plt.clf()
    if plot:
        initial_plot(plt, centroids, dataset, pointLabels)
    print('Initial infomation:')
    print('\tStopping Criteria: ', stoppingCriteria.__name__)
    print('\tMax Iterations: ', str(max_iter))
    print('\tDistance Function: ', distanceFunction.__name__)
    print('\tInitial Centroids:', centroids)

    # Set defualt values and start timer
    new_centroids = [[-1,-1],[-1,-1]]
    clusters = [[] for i in range(numClusters)]
    sse = [float('inf'),float('inf') ]
    iteration = 0
    run = True
    if timer:
        start_time = time.time()

    while run:        
        # Increase iteration
        iteration = iteration + 1

        # Reset clusters
        for i in range(numClusters):
            clusters[i] = []
        # Assign data points to clusters based on passed distance function
        clusters = assign(dataset, centroids, clusters, distanceFunction)

        # Recalculate centroid locations
        new_centroids = []
        new_centroids = recalculate_cluster(clusters,len(dataset[0])-1)

        # Calcualte the current SSE value
        sse[1] = calculate_sse(new_centroids,clusters)

        # Print the iteration if asked
        if iteration in printIteration:
                # Print iteration 1 values
                print('Iteration ' + str(iteration) + ':')
                print('Current SSE:', sse[1])
                print(new_centroids)
                clusternames = []
                if clusterName:
                    # Determine name by majority vote and accuracy
                    name, accuracy, correct, incorrect  = cluster_info(cluster, i)
                    clusternames.append(name)
                if plot:
                    plot_clusters(plt, new_centroids, clusters, clusternames, pointLabels)
        
        # Special case for the first iteration, ensure that the first sse is smaller than the compared sse
        if iteration == 1:
            sse[0] = sse[1] + 1

        # Determine if stopping criteria has been reached
        run = stoppingCriteria(centroids,new_centroids,iteration,max_iter,sse)
        
        # Set current centroid to new centroid
        centroids = new_centroids
        # Set current SSE to new SSE
        sse[0] = sse[1]

    # End time before printing results
    if timer:
        end_time = time.time()
        print('\nExecution time: ', end_time - start_time)

    # Print final values
    print('Final Output:')
    print('\tFinal SSE:', sse[1])
    print('\tNumber of iterations: ' + str(iteration))
    print('\tFinal Centroids:',centroids)
    print('\tFinal Clusters: saved in Clusters.txt','\n\n')
    f = open('Clusters.txt', 'a')
    f.write('Clusters for ' + distanceFunction.__name__ + '\n')
    clusternames = []
    for i,cluster in enumerate(clusters):
        if clusterName:
            # Determine name by majority vote and accuracy
            name, accuracy, correct, incorrect  = cluster_info(cluster, i)
            clusternames.append(name)
            # Print Name and accuracy statements
            f.write('Cluster: ' + name + '\n')
            f.write('Accuracy: ' + str(accuracy) + '\t Correct: ' + str(correct) + '\tIncorrect: ' + str(incorrect) + '\n')
            print('Cluster: ' + name)
            print('Accuracy: ' + str(accuracy) + '\t Correct: ' + str(correct) + '\tIncorrect: ' + str(incorrect),'\n')
        else:
            f.write('Cluster ' + str(i+1) + '\n')

        for line in cluster:
            f.write(str(line))
            f.write('\n')
        f.write('\n')    
    f.close()
    if plot:
        plot_clusters(plt, centroids, clusters, clusternames, pointLabels)

############################################################
#   Tasks
############################################################
def task_1():
    initial_centroids = [[[4,6],[5,4]],[[4,6],[5,4]],[[3,3],[8,3]],[[3,2],[4,8]]]
    points = [(3,5,'X1'),(3,4,'X2'),(2,8,'X3'),(2,3,'X4'),(6,2,'X5'),(6,4,'X6'),(7,3,'X7'),(7,4,'X8'),(8,5,'X9'),(7,6,'X10')]

    for i in range(len(initial_centroids)):
        print('=======================================')
        print('Question ' + str(i+1))
        print('=======================================')
        if i == 1:
            kmeans(points,2,euclidean,initCentroids=initial_centroids[i],printIteration=[1])
        else:
            kmeans(points,2,manhattan,initCentroids=initial_centroids[i],printIteration=[1])

def task_2():
    data = pd.read_csv('iris.data')
    points = data.values.tolist()

    # Create random but consistent initial centroids
    random.seed(time.time())
    data = []
    for i in range(len(points)):
        line = []
        for j in range(len(points[0])-1):
            line.append(points[i][j])
        data.append(line)
    centroids = random.sample(data, 3)

    print('=======================================')
    print('Question 1 & 2 & 3')
    print('=======================================')
    f = open('Clusters.txt', 'a')
    f.write('=======================================\n')
    f.write('Question 1 & 2 & 3\n')
    f.write('=======================================\n')
    f.close()
    distances = [euclidean,cosine,jaccard]
    for distance in distances:
        print('Kmeans for ' + distance.__name__ + '...')
        kmeans(points,3,distance,plot=True,clusterName=True,initCentroids=centroids,pointLabels=False,timer=True)

    print('=======================================')
    print('Question 4')
    print('=======================================')
    f = open('Clusters.txt', 'a')
    f.write('=======================================\n')
    f.write('Question 4\n')
    f.write('=======================================\n')
    f.close()
    modes = [centroid_based, sse_based, iteration_based]
    for mode in modes:
        kmeans(points,3,euclidean,stoppingCriteria=mode,max_iter=100,plot=False,initCentroids=centroids,timer=True)
        kmeans(points,3,cosine,stoppingCriteria=mode,max_iter=100,plot=False,initCentroids=centroids,timer=True)
        kmeans(points,3,jaccard,stoppingCriteria=mode,max_iter=100,plot=False,initCentroids=centroids,timer=True)

############################################################
#   Main
############################################################
task_1()  
task_2()