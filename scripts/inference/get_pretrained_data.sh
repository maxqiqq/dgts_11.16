mkdir -p checkpoints
cd checkpoints
wget https://www.dropbox.com/s/cj4godit56s2gpe/pretrained_wts.tar.gz
tar -xf pretrained_wts.tar.gz && rm pretrained_wts.tar.gz
#是直接用的Dropbox已有数据吗？咋感觉和上面的train过程没关系？

mkdir -p ../dataset/raw
cd ../dataset
wget https://www.dropbox.com/s/7krbgcibzxifgcc/inference_data.tar.gz
tar -xf inference_data.tar.gz && rm inference_data.tar.gz
mv _raw/* raw/  #mv(move and rename files)
rm -rf _raw