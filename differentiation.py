# FY1006 module - maths/computing

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = sp.Symbol('x')

# FY1006: Using sympy in python for differentiation
# Make sure you look at the video for a walkthrough of how to do the following questions
# Then come back here and start practicing :)

# Q1: differentiate the following equation (i.e. get f'(x))
# f(x) = 3x^5 - 10x^2 + 21

def f1(x):
    return (3)*x**(5) - (10)*x**(2) + 21

print(sp.diff(f1(x)))

# Q2: For the equation:
# f(x) = (2x^3)/(10x)
# Do the following:
# 1) Differentiate f(x)

def f2(x):
    return ((2)*x**(3))/((10)*x)

print(sp.diff(f2(x)))

# 2) Calculate the tangent line at x = 4

diff_equat2 = sp.diff(f2(x))
diff_func2 = sp.lambdify(x, diff_equat2)

print(diff_func2(4))

# now finding y = mx +c (aka. equation of tangent line)

grad = diff_func2(4)
c_val = f2(4) - grad*4

print(c_val)

def tangent_equat(x):
    return grad*x + c_val

print(tangent_equat(x))

# 3) Using values of x between 1 and 10 (1 <= x <= 10) plot f(x) and the tangent at x = 4

x_val_q2 = np.arange(1, 10, 0.5)
y_val_q2 = f2(x_val_q2)
tangent_line = tangent_equat(x_val_q2)

plt.plot(x_val_q2, y_val_q2, label="f(x)")
plt.plot(x_val_q2, tangent_line, label="tangent at x = 4")

plt.legend()
plt.show()

# Q3 - Using sympy to differentiate f(x) to f'(x)
#     Plot both f(x) and f'(x) on a graph
#  f(x) = 3 sin (2x)

def sin_func(x):
    return 3*(sp.sin(2*x))

def sin_func_yval(x):
    return 3*(np.sin(2*x))

diff_equat3 = sp.diff(sin_func(x))
diff_func3 = sp.lambdify(x, diff_equat3)

x_val_q3 = np.arange(-5, 5, 0.1)
y_val_q3 = sin_func_yval(x_val_q3)
y_val_diff = diff_func3(x_val_q3)

plt.plot(x_val_q3, y_val_q3, label="f(x)")
plt.plot(x_val_q3, y_val_diff, label="f'(x)")

plt.axhline(color="k")
plt.axvline(color="k")
plt.legend()
plt.show()


# Note on getting the values to plot:
# To get my y_values, I used a separate function using np.sin instead of sp.sin_
# Using sp.diff and sp.lambdify worked fine for getting my f' values_
# sp.sin doesn't seem to like taking a numpy array_

# Q4 - For the following equation: f(x) = (1/4)x^4 - 2x^2
# # 1) Find the critical points (where f'(x) = 0)
#
def f4(x):
    return (1/4)*x**(4) - 2*x**2

diff_equat4 = sp.diff(f4(x))
diff_func4 = sp.lambdify(x, diff_equat4)

print(f"f'(x) = {diff_equat4}")

# find points where f'(x) = 0

crit_points = sp.solveset(sp.Eq(diff_equat4, 0), x)

print(f" f(x) has {crit_points} of critical points.")

# determining whether they are maxima or minima

for point in crit_points:
    if diff_func4(point-0.1) > 0 and diff_func4(point+0.1) < 0:
        print(f" {point} is a critical maxima. ")
        plt.plot(point, f4(point), 'b*')
    elif diff_func4(point-0.1) < 0 and diff_func4(point+0.1) > 0:
        print(f" {point} is a critical minima. ")
        plt.plot(point, f4(point), 'r^')

# 2) Plot a graph of f(x) for 2.5 <= x <= 2.5

x_val_q4 = np.arange(-2.5, 2.5, 0.1)
y_val_q4 = f4(x_val_q4)

plt.plot(x_val_q4, y_val_q4, label="f(x)")
plt.axhline(color="k")
plt.axvline(color="k")
plt.legend()
plt.show()

# 3) Plot each critical point on the graph, with a red triange (r^) for minima
# and a blue star (b*) for maxima
# Hint: You should be able to use a for loop, and an if/else statement, to determine if a
# critical point is a minima or maxima_


# Q5 - for the following equation:
# f(x) = 8x^21 + 5x^5 - sqrt(x)
# 1) Find f'(x)
# 2) Find f''(x)
# 3) Find f'''(x)

def f(x):
    return 8*x**(21) + 5*x**(5) - sp.sqrt(x)

first_deriv = (sp.diff(f(x), x))
second_deriv = (sp.diff(f(x), x, x))
third_deriv = (sp.diff(f(x), x, x, x))

print(f" f'(x) = {first_deriv}")
print(f" f''(x) = {second_deriv}")
print(f" f'''(x) = {third_deriv}")


# Q6: Use implicit differentiation to calculate dy/dx for the following equation:
# f(x) = 3x^2 + 2xy + y^2

x = sp.Symbol('x')
y = sp.Symbol('y')


def implicit_diff_func(x, y):
    return 3*x**(2) + 2*x*y + y*(2)


print(sp.idiff(implicit_diff_func(x, y), y, x))



# Q7: For the following equation:
# f(x) = (2/5)x^5 - x^3 + 10x
# Do the following:
# 1) Differentiate f(x)

def f7(x):
    return (2/5)*x**5 - x**3 + 10*x

print(sp.diff(f7(x), x))

# 2) Plot f(x) for 1 <= x <= 20

x_val_q7 = np.arange(1, 20, 1)
y_val_q7 = f7(x_val_q7)

# 3) Plot a tangent to the curve at x = 10

diff_equat7 = sp.diff(f7(x))
diff_func7= sp.lambdify(x, diff_equat7)

grad = diff_func7(10)
c_val = f7(10) - grad*10

def tangent_equat2(x):
    return grad*x + c_val

tangent_line2 = tangent_equat2(x_val_q7)

plt.plot(x_val_q7, y_val_q7, label="f(x)")
plt.plot(x_val_q7, tangent_line2, label="tangent at x = 10")
plt.legend()
plt.show()

# Q8: For the following equation:
# f(x) = 3sin(2x) - 2cos(3x)
# Do the following:
# 1) Find f'(x)

def trig_deriv(x):
    return (3*(sp.sin(2*x))) - (2*(sp.cos(3*x)))

def trig_func_yval(x):
    return (3*(np.sin(2*x))) - (2*(np.cos(3*x)))

first_deriv_eq = sp.diff(trig_deriv(x), x)
first_deriv_fc = sp.lambdify(x, first_deriv_eq)
print(first_deriv_eq)

# 2) Find f''(x)

second_deriv_eq = sp.diff(trig_deriv(x), x, x)
second_deriv_fc = sp.lambdify(x, second_deriv_eq)
print(second_deriv_eq)

# 3) Plot f(x), f'(x) and f''(x) on a graph

x_val_q8 = np.arange(-5, 5, 0.1)
y_val_q8 = trig_func_yval(x_val_q8)
y_val_first_deriv = first_deriv_fc(x_val_q8)
y_val_second_deriv = second_deriv_fc(x_val_q8)

plt.plot(x_val_q8, y_val_q8, label="f(x)")
plt.plot(x_val_q8, y_val_first_deriv, label="f'(x)")
plt.plot(x_val_q8, y_val_second_deriv, label="f''(x)")

plt.axhline(color="k")
plt.axvline(color="k")
plt.legend()
plt.show()

# Q9: Differentiate the following equation:
# f(x) = (1/2)exp(5x)

def exp(x):
    return (1/2)*sp.exp(5*x)

print(sp.diff(exp(x), x))


# Q10: Differentiate the following equation:
# f(x) = 2log10(x)

def log_base_10(x):
    return (2)*sp.log(x,10)

print(sp.diff(log_base_10(x)))

# Well done for making it to the end of today's worksheet :) ls below, feel free to practice with any other differentiation problems

