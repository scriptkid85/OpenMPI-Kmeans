#Number of Points
b=10
#Number of Cluster
k=2
#Number of dimensions
v=10
echo ********GENERATING $b INPUT POINTS EACH IN $k CLUSTERS 
python ./randomclustergen/genDNA.py -c $k -p $b -o input/clusterDNA.csv -v $v
