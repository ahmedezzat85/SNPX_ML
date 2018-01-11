cd
git clone https://github.com/ahmedezzat85/SNPX_ML.git
sudo chmod 777 SNPX_ML -R
cd SNPX_ML/python

# Download the dataset
./cifar10_download.sh

source activate tensorflow_p36
./train.sh