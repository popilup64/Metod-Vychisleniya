import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**3 - 6*x**2 + 5*x

def f_proiz(x):
    return 3*x**2 - 12*x + 5

a = -1
b = 6

x = np.linspace(a, b, 100)
y = f(x)
y1 = 0*x
plt.figure(1)
plt.plot(x, y, color = 'green')
plt.plot(x, y1, color = 'red')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('График функции')

def bisection(a, b, eps):
    count_iter = 0
    while abs(b-a) > 2*eps:
        count_iter += 1
        medl = (a+b)/2
        if (f(a) * f(medl)) < 0:
            b = medl
        else:
            a = medl
    root = (a+b)/2
 
    return root, count_iter

def xord(a, b, eps):
    count_iter = 0
    x_prev = a

    while True:
        count_iter += 1
        x_new = a - f(a)*(b-a)/(f(b)-f(a))
        if (f(a) * f(x_new) < 0):
            b = x_new
        else:
            a = x_new
        if (abs(x_new - x_prev) <= eps):
            return x_new, count_iter
        
        x_prev = x_new

def nuton (a, b, eps):
    x_prev = (a+b)/2
    iter_count = 0

    while True:
        x_new = x_prev - f(x_prev)/f_proiz(x_prev)
        if (abs(x_new - x_prev) <= eps):
            return x_new, iter_count
        x_prev = x_new
        iter_count += 1


def plot_graf(a, b, color, linestyle_dict, figure_num, log_style):
    eps_val = np.logspace(-5, 0, 5)
    iter_val1 = []
    iter_val2 = []
    iter_val3 = []

    for eps in eps_val:
        _, iter_count1 = bisection(a, b, eps)
        iter_val1.append(iter_count1)
        _, iter_count2 = xord(a, b, eps)
        iter_val2.append(iter_count2)
        _, iter_count3 = nuton(a, b, eps)
        iter_val3.append(iter_count3)

    if log_style:    
        plt.figure(figure_num)
        plt.semilogx(eps_val, iter_val1, color=color, linestyle=linestyle_dict['bisection'], 
                    linewidth=2, label=f'Бисекция [{a}, {b}]')
        plt.semilogx(eps_val, iter_val2, color=color, linestyle=linestyle_dict['chord'], 
                    linewidth=2, label=f'Хорды [{a}, {b}]')
        plt.semilogx(eps_val, iter_val3, color=color, linestyle=linestyle_dict['newton'], 
                    linewidth=2, label=f'Ньютон [{a}, {b}]')
    
    else:
        plt.figure(figure_num)
        plt.plot(eps_val, iter_val1, color=color, linestyle=linestyle_dict['bisection'], 
                    linewidth=2, label=f'Бисекция [{a}, {b}]')
        plt.plot(eps_val, iter_val2, color=color, linestyle=linestyle_dict['chord'], 
                    linewidth=2, label=f'Хорды [{a}, {b}]')
        plt.plot(eps_val, iter_val3, color=color, linestyle=linestyle_dict['newton'], 
                    linewidth=2, label=f'Ньютон [{a}, {b}]')

linestyle_dict = {'bisection': '-', 'chord': '--', 'newton': ':'}

eps = 1e-3
# лог шакала
plt.figure(2)
plot_graf(-3, 0.5, 'red', linestyle_dict, 2, True)  
plot_graf(0.5, 3, 'green', linestyle_dict, 2, True)   
plot_graf(3, 9, 'blue', linestyle_dict, 2, True)     
plt.grid()
plt.xlabel('eps')
plt.ylabel('Итерации')
plt.title('График (логарифмическая шкала)')
plt.legend()


# обычная шкала
plt.figure(3)
plot_graf(-3, 0.5, 'red', linestyle_dict, 3, False)  
plot_graf(0.5, 3, 'green', linestyle_dict, 3, False)   
plot_graf(3, 9, 'blue', linestyle_dict, 3, False)      
plt.grid()
plt.xlabel('eps')
plt.ylabel('Итерации')
plt.title('График (обычная шкала)')
plt.legend()


root1_bis = bisection(-3, 0.5, eps)
root1_xord = xord(-3, 0.5, eps)
root1_nuton = nuton(-3, 0.5, eps)

print("Корни на интервале [-3, 0.5]:")
print(f"Бисекция: {root1_bis[0]}, Итерации: {root1_bis[1]}")
print(f"Хорды: {root1_xord[0]}, Итерации: {root1_xord[1]}")
print(f"Ньютон: {root1_nuton[0]}, Итерации: {root1_nuton[1]}")
print()

root2_bis = bisection(0.5, 3, eps)
root2_xord = xord(0.5, 3, eps)
root2_nuton = nuton(0.5, 3, eps)

print("Корни на интервале [0.5, 3]:")
print(f"Бисекция: {root2_bis[0]}, Итерации: {root2_bis[1]}")
print(f"Хорды: {root2_xord[0]}, Итерации: {root2_xord[1]}")
print(f"Ньютон: {root2_nuton[0]}, Итерации: {root2_nuton[1]}")
print()

root3_bis = bisection(3, 9, eps)
root3_xord = xord(3, 9, eps)
root3_nuton = nuton(3, 9, eps)

print("Корни на интервале [3, 9]:")
print(f"Бисекция: {root3_bis[0]}, Итерации: {root3_bis[1]}")
print(f"Хорды: {root3_xord[0]}, Итерации: {root3_xord[1]}")
print(f"Ньютон: {root3_nuton[0]}, Итерации: {root3_nuton[1]}")
print()

plt.show()