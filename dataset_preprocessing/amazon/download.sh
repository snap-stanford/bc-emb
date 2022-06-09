for category in Video_Games Musical_Instruments Grocery_and_Gourmet_Food
do
mkdir -p files/${category}/raw
wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_${category}.json.gz
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/${category}_5.json.gz
mv meta_${category}.json.gz files/${category}/raw
mv ${category}_5.json.gz files/${category}/raw
done
