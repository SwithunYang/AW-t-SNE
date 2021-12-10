# AW-t-SNE
The test data, code and result of the AW t-SNE algorithm

# Structure of the folder
## Datasets: This folder contains two datasets, the MNIST dataset and the medical record dataset.
### MNIST dataset:We selected two easily misidentified digits, 4 and 9, based on the digit labels to form the new dataset(mnist_data(1000) and mnist_data(2000)):
    mnist_data(1000)：mnist_data(1000) contains a total of 1000 groups of data, each consisting of 784 data point attributes and 1 label attribute.
    mnist_data(2000)：mnist_data(2000) contains a total of 1000 groups of data, each consisting of 784 data point attributes and 1 label attribute.
    handled_mnist_data(1000)：the dataset obtained by binarising mnist_data(1000).
    handled_mnist_data(2000)：the dataset obtained by binarising mnist_data(2000).
    
### Medical Record dataset:A total of 13 attributes are included:
    medical record_data：This is our origin medical record data, which contains 13 attributes, namely LOH, COG, AGE, TOO, SDH, LSH, CEH, TCH, CWM, COD, CLH, CCT, CCM
    handled_medical record_data(1000)：The dataset is obtained by standardising the medical record_data and randomly selecting 1000 groups of data.
    handled_medical record_data(2000)：The dataset is obtained by standardising the medical record_data and randomly selecting 2000 groups of data.
   
## Module：This folder contains all the codes that needs to be used.
    critic method.py：This is the code for calculating the weights of the data matrix by the critic weight method.
    svd method.py：This is the code for calculating the weights of the data matrix by the svd method.
    entrophy weight method.py：This is the code for calculating the weights of the data matrix by the entrophy weight method.
    PSO(MNIST).py：This is the code to calculate the optimal weights of the MNIST data matrix by the PSO algorithm.
    PSO(Medical Record).py：This is the code to calculate the optimal weights of the Medical Record data matrix by the PSO algorithm.
    t-SNE(MNIST).py：This is the code for the t-SNE dimensionality reduction algorithm of MNIST data.
    t-SNE(medical record).py：This is the code for the t-SNE dimensionality reduction algorithm of Medical Record data.
    AW t-SNE(Medical Record).py：This is the code for the t-SNE algorithm for dimensionality reduction of Medical Record data.
    AW t-SNE(MNIST).py：This is the code for the t-SNE algorithm for dimensionality reduction of MNIST data.
    
