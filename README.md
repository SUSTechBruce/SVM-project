# SVM-project
Using two basic method, one is SMO algorithm, the other one is Gradient descent algorithm, to deal with ten dimensional data set.


The classical algorithms in the SVM problem are the gradient descent method and the convex quadratic programming algorithm. In this lab, I use John Platt's SMO standard algorithm. The sequence minimum optimization problem is to convert the optimization problem into multiple small problems, in order to find the value of alpha and intercept b. The core idea of this algorithm is to find two alpha values in each loop. They satisfy two conditions. One is that the two alphas must be outside the interval boundary, and the other two aphas values
are not subject to standard interval processing. Or not on the border. The best alphas pair to beoptimized is then determined in the outer loop using the standard SMO algorithm. After completing the SMO algorithm, I used the pegasos stochastic gradient descent algorithm. The
core of the algorithm is to randomly select a training sample to calculate the gradient of the objective function and then go a specific step in the opposite direction. The complexity of the overall algorithm is O(n/λε), and n is the dimension of the data set. The algorithm can be optimized to O(d/ελ).
