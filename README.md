# PAforASD
Project on PA
1. In this ipynb file i implement K-means clustering algorithms to MNIST dataset. MNIST dataset has 70000 images of hand-written digits from 0 to 9. 
K-means clustering is an unsupervised learning algorithm that groups unlabeled dataset into different clusters. Here are the key steps:
1) Define the number of clusters, denoted as “k.”
2) Randomly initialize k vectors, serving as the initial cluster centers.
3) Calculate the distance between each data point and every cluster center.
4) Assign each data point to the cluster center it is closest to.
5) Recalculate the cluster centers by finding the mean of each new cluster.
6) Repeat steps 3–5 for a specified number of iterations or until convergence.

Here i implement naive k-means clustering and a version with parallelization. I perform 100 iterations for each version and calculate mean time per iteration, then i compare this time and calculate speedup made by parallelization.

2. The results can be easily reproduced by running code and installing packages that are listed there such as: 
- matplotlib.pyplot  
- numpy     
- torch
- time
- tqdm.notebook
- sklearn.datasets 
Also the code should be implemented on GPU.
3. Parallelization is performed by GPU acceleration with PyTorch to speed up the k-means clustering algorithm, instead of CPU. Also optimized version utilizes torch for tensor operations instead of numpy arrays, matrix operations and broadcasting to calculate distances efficiently instead of nested loops, matrix multiplication and division to update cluster centers efficiently intead of nested loops. 

4. ![image](https://github.com/user-attachments/assets/03fe563f-eec1-47c1-a430-93c22c1cd01a)
Here we can see that after using parallelization the time lowered a lot, specifically in 904 times. Naive algorithm took 9.79 sec per iteration, while updated algorithm took 0.01 sec per iteration.

