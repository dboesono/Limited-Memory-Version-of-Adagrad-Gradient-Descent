import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from numpy import asarray
from numpy import arange
from numpy import meshgrid
from matplotlib.pyplot import figure


def objective(x):
    # The objective(x) function takes on a 2d array as its parameter.
    return x[0]**4 + (2*x[0]**3)/3 + x[0]**2/2 - 2*x[0]**2*x[1] + 4*x[1]**2/3


def gradient(x):
    # The gradient function computes the partial derivative of each variable and store it into a array.
    grad = list()
    dfdx = 4*x[0]**3 + 2*x[0]**2 + x[0] - 4*x[0]*x[1]
    dfdy = (8/3)*x[1] - 2*x[0]**2
    grad.append(dfdx)
    grad.append(dfdy)
    return np.array(grad)


def l2_norm(x):
    # The l2_norm function computes the L2 norm of a 2 dimension vector.
    return (x[0]**2 + x[1]**2)**0.5


def backtracking(gamma, sigma, xk, dk):
    """
    The backtracking function Minimize over the alpha from the function f(xk + alpha*dk) and alpha > 0 is assumed to
    be a descent direction

    Parameters
    ----------
    xk: A current point from the function that takes the form of an array.

    dk: The search or descent direction usually takes the value of the first
        derivative array of the function.

    sigma: The value of the alpha shrinkage factor that takes the form of a float.

    gamma: The value to control the stopping criterion that takes the form of a float.

    Returns
    -------
    alpha: The value of alpha at the end of the optimization that takes the form of a scalar.

    """
    # Initialize the alpha equal to the value 1
    alpha = 1

    # Began the procedure based on the condition in the while loop
    while True:
        if objective(xk + alpha * dk) <= objective(xk) + gamma * alpha * np.inner(dk, gradient(xk)):
            return alpha
        else:
            alpha = alpha * sigma


def gradient_descent_adagrad(init, tol, scale, memo, gamma, sigma):
    """
    Adagrad Gradient descent method for unconstraint optimization problem given a starting point x which is a element
    of real numbers. The algorithm will repeat itself accoding to the following procedure:

    1. Define the descending direction using adaptive diagonal scaling.
    2. Using a step size strategy, choose the step length alpha using the Armijo Line Search/Backtracking strategy.
    3. Update the x point using the formula of x := x + alpha*direction

    Repeat this procedure until a stopping criterion is satisfied.

    Parameters
    ----------
    init: The initial value of x and y in the form of an array

    tol: The tolerance for the l2 norm of f_grad

    scale: The scaling parameter used for the adaptive diagonal scaling process

    memo: The memory Parameter that will limit the summation in the adaptive diagonal scaling process

    gamma: The value to control the stopping criterion that takes the form of a float

    sigma: The value of the alpha shrinkage factor that takes the form of a float.

    Returns
    -------
    Solutions: The vector of the coordinates in the learning path.

    values: The value of the objective function along the learning path.

    """

    # Initialize the initial point x as an array and two array to store the x and f(x) values
    # and the number of iteration
    xk = np.array(init)
    curve_x = []
    curve_y = []
    num_iter = 1
    print('Initial Condition: y = {}, x = {} \n'.format(objective(xk), xk))

    # Utilize the stopping criterion when the L2 norm is smaller than the tolerance value as the condition
    # in the while loop procedures
    while l2_norm(gradient(xk)) > tol:

        # Append the initial point and initial objective value in the arrays storing x and f(x) values
        curve_x.append(xk)
        curve_y.append(objective(xk))

        # Define the tm(k) function which is tm(k) = max{0, k-m}
        tmk = max(0, num_iter - memo)

        # Adaptive diagonal scaling  based on memory parameter
        D1 = scale
        D2 = scale
        for j in range(tmk, num_iter):
            D1 += (gradient(curve_x[j])[0]) ** 2
            D2 += (gradient(curve_x[j])[1]) ** 2
        D1 = D1 ** 0.5
        D2 = D2 ** 0.5

        # Storing vk values into the Dk array as their reciprocals
        Dk = np.array(1 / D1, 1 / D2)

        # Define the descending direction
        dk = -1 * np.multiply(Dk, gradient(xk))

        # Find an suitable step size or alpha
        alpha = backtracking(gamma, sigma, xk, dk)

        # Update the x value using the gradient descent algorithm equation
        xk = xk + alpha * dk

        # Update the number of iterations and print out the results from the iteration
        num_iter += 1
        print('Iteration: {} \t y = {}, x = {}, gradient = {:.4f}'.format(num_iter - 1, objective(xk), xk,
                                                                          l2_norm(gradient(xk))))

    # Append the final results to their respective arrays
    curve_x.append(xk)
    curve_y.append(objective(xk))

    # Print out the final results of the algorithm
    print('\nSolution: \t y = {}, x = {}'.format(objective(xk), xk))

    # Return the x and f(x) values in array forms
    return curve_x, curve_y


# Initiate algorithm using the selected parameters and different initial points
solution, value = gradient_descent_adagrad([3,3], 10**(-5), 10**(-6), 25, 0.1, 0.5)
solution1, value1 = gradient_descent_adagrad([-3,3], 10**(-5), 10**(-6), 25, 0.1, 0.5)
solution2, value2 = gradient_descent_adagrad([3,-3], 10**(-5), 10**(-6), 25, 0.1, 0.5)
solution3, value3 = gradient_descent_adagrad([-3,-3], 10**(-5), 10**(-6), 25, 0.1, 0.5)


# Plot the learning path of the adagrad gradient descent algorithm in a contour plot
bounds = asarray([[-4.0, 4.0], [-4.0, 4.0]])
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute the targets
results = objective([x,y])
# create a filled contour plot with 50 levels and jet color scheme
plt.contourf(x, y, results, levels=50, cmap='jet')
# plot the sample as black circles
solution = asarray(solution2)
plt.plot(solution[:, 0], solution[:, 1], '.-', color='w')
# show the plot
plt.rcParams["figure.figsize"] = (24,6)
plt.tick_params(axis='y', labelsize=20)
plt.tick_params(axis='x', labelsize=20)
plt.title('Adagrad Gradient Descent', fontsize=20)
plt.show()
