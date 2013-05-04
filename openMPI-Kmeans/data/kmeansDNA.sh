#Number of Points
b=1000000
#Number of Cluster
k=3
#Number of dimensions
v=10
echo ********GENERATING $b INPUT POINTS EACH IN $k CLUSTERS 
python ./randomclustergen/genDNA.py -c $k -p $b -o input/clusterDNA.big.csv -v $v
