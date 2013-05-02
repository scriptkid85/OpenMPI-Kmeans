#Number of Points
b=10
#Number of Cluster
k=2
echo ********GENERATING $b INPUT POINTS EACH IN $k CLUSTERS 
python ./randomclustergen/generaterawdata.py -c $k -p $b -o input/cluster.csv
