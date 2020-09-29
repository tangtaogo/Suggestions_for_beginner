import os
import tarfile
from scipy import io
import shutil

# train
def un_tar(file_name, output_root='train'):
    # untar zip file to folder whose name is same as tar file
    tar = tarfile.open(file_name)
    names = tar.getnames()

    file_name = os.path.basename(file_name)
    extract_dir = os.path.join(output_root, file_name.split('.')[0])
    
    # create folder if nessessary
    if os.path.isdir(extract_dir):
        pass
    else:
        os.mkdir(extract_dir)

    for name in names:
        tar.extract(name, extract_dir)
    tar.close()


def untar_traintar(traintar='./ILSVRC2012_img_train'):
    """
    untar images from traintar and save in corresponding folders
    organize like:
    /train
       /n01440764
           images
       /n01443537
           images
        .....
    """
    root, _, files = next(os.walk(traintar))
    for file in files:
        un_tar(os.path.join(root, file))



if __name__ == '__main__':
    untar_traintar()
    
