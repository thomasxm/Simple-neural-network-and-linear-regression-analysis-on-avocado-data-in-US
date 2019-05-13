import numpy as np


class LinearRegression(object):
    def __init__(self, X, y, theta, lambdaa):
        # Set number of nodes in input, hidden and output layers.
        self.X = X
        self.y = y
        self.theta = theta
        self.lambdaa = lambdaa


    def linearRegCostFunction(self, X, y, theta, lambdaa):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        # Initialize some useful values
        m = np.size(y); # number of training examples
        n = np.size(theta);

        # You need to return the following variables correctly 
        J = 0;
        grad = np.zeros(np.size(theta)).reshape(n, 1, order='F');

        
        h = np.matmul(X, theta);

        
        ### TO prevent y and h from being the broadcasting in the wrong way as: (x,)+(x,1) = (x,x)
        ### instead of (x+x,1)
        if(y.shape == (y.shape[0],)):
            y = y[:, np.newaxis]  
        if(h.shape == (h.shape[0],)):
            h = h[:, np.newaxis]  
        hError = (h - y);       

        #sumSquaredError = np.matrix.sum(np.power(hError, 2), 1);
        sumSquaredError = np.sum(np.power(hError, 2));

        regTermLeft = (1/(2 * m)) * sumSquaredError;

        thetaWithoutBias = theta[1:];
        sumSquaredTheta = np.sum(np.power(thetaWithoutBias, 2));
        regTermRight = (lambdaa / (2 * m)) * sumSquaredTheta;

        J = regTermLeft + regTermRight;
        
        #grad = (1/m) * X' * hError;
        #grad = (1/m) * np.matmul(X.transpose(),hError.reshape(m,-1));
        grad = (1/m) * np.matmul(X.transpose(),hError);


        #print((lambdaa / m) * thetaWithoutBias.reshape(len(thetaWithoutBias),-1))
        grad[1:] += (lambdaa / m) * thetaWithoutBias;
        #grad = grad.flatten('F');
        
        return J, grad

    
    def gradientDescentMulti(self, X, y, theta, alpha, lambdaa, num_iters):

        m = np.size(y); # number of training examples
        J_history = np.zeros((num_iters, 1));
        n = X.shape[1]; # number of features
        #theta = theta.reshape(n, 1, order='F');
     
        for iter in range(num_iters):

    
            #theta = theta - alpha .* (1./m) .* ((X*theta - y)' * X)';  
            #predictions =  np.matmul(X , theta);
            #print(np.size(predictions))
            #print(np.size((np.substeact(predictions, y)))
            #updates = np.matmul( X.transpose(), (predictions - y) );
            #updates = np.matmul( X.transpose(), (predictions - y) );
            #theta = theta - alpha * (1/m) * updates;


            J_history[iter], grad = self.linearRegCostFunction(X, y, theta, lambdaa);
            #print("the shape of gradient:")
            #print(grad.shape)
            #print("the shape of theta before uodate:")
            #print(theta.shape)
            theta = theta.reshape(n,-1) - alpha * grad;
            #print("the shape of theta after uodate:")
            #print(theta.shape)
            

        return theta, J_history
    

# #########################################################
# # Set your hyperparameters here
# ##########################################################
# iterations = 5500
# learning_rate = 1
# hidden_nodes = 10
# output_nodes = 1
