mkdir -p dataset  #mkdir创建目录，-p(parents)表示上层目录，后面dataset是名称
cd dataset  #指定要进入的目录
wget https://www.dropbox.com/s/7crbp0c8irl3gs0/dgts_train_data.tar.gz  #文件下载
tar -xvf dgts_train_data.tar.gz && rm dgts_train_data.tar.gz
#tar-xvf是提取文件/解压；&&左边命令被执行成功后再执行右边；rm(remove files and directories)删除原来的压缩文件
