# image-captioning

##### Overview ######
Artificial intelligence project on image captioning using the MS COCO 2017 dataset, combining RNN and CNN architectures.

##### Setup #####
The setup is only required if you wish to train the model yourself.

To get started, run the 'dataset_loader.py' file located in the 'dataset' folder to download the MS COCO 2017 dataset to your PC. Ensure you have sufficient storage, as the entire dataset is 26GB, but at least 40GB of free space is required for proper extraction. The dataset is first downloaded as .zip files, which are then extracted before the archives are deleted.

If you encounter issues with 'wget', it means that `wget` is not installed on your system.  
To fix this:  

1. Download `wget.exe` from [this link](https://eternallybored.org/misc/wget/).  
2. Add it to your system `PATH`.  
   - On Windows, you can simply place it in `C:\Windows\System32` (already in the `PATH`).  

This will allow you to retrieve files using HTTP, HTTPS, and FTP protocols.  