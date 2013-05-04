#Number of Points
b=10000
#Number of Cluster
k=5
echo ********GENERATING $b INPUT POINTS EACH IN $k CLUSTERS 
python ./randomclustergen/generaterawdata.py -c $k -p $b -o input/cluster.big.csv
