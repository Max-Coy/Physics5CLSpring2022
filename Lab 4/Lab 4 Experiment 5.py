import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import math


def sample_stdev(data):
    return np.std(data, ddof = 1, axis = -1)


def standard_error(data):
    return sample_stdev(data) / np.sqrt(len(data) if data.ndim == 1 else len(data[0]))


def data_statistics(data):
    return np.mean(data), sample_stdev(data), standard_error(data)


def covariance(x, y):
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    covariance = 0
    for i in range(len(x)):
        covariance += (x[i] - x_bar) * (y[i] - y_bar)

    return covariance / (len(x) - 1.0)


def correlation_coefficient(x, y):
    return covariance(x, y) / np.sqrt((covariance(x, x) * covariance(y, y)))


def chi_squared(predicted, observed, errors):
    res = observed - predicted
    norm_res = res / errors
    return np.sum(np.multiply(norm_res, norm_res))


class WeightedLinearFit(object):
    def __init__(self, x, y, alpha = None):
        self.x = x
        self.y = y
        self.alpha = alpha

        self.has_optimized = False

        self.linear_model = None
        self.m = 0
        self.c = 0
        self.m_err = 0
        self.c_err = 0

    @staticmethod
    def linear_fit(x, m, c):
        return m * x + c

    def optimize(self, guess = [0, 0]):
        if self.alpha is not None:
            self.linear_model = opt.curve_fit(WeightedLinearFit.linear_fit, self.x, self.y, sigma = self.alpha, p0 = guess)
        else:
            self.linear_model = opt.curve_fit(WeightedLinearFit.linear_fit, self.x, self.y)
        self.m = self.linear_model[0][0]
        self.c = self.linear_model[0][1]
        self.m_err = np.sqrt(self.linear_model[1][0][0])
        self.c_err = np.sqrt(self.linear_model[1][1][1])
        self.has_optimized = True

    def calculate(self, x):
        if self.has_optimized:
            return self.m * x + self.c

        raise Exception('Need to optimize model first.')


class DirectFit(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.has_optimized = False

        self.linear_model = None
        self.m = 0
        self.m_err = 0

    @staticmethod
    def linear_fit(x, m):
        return m * x

    def optimize(self, guess = [0]):
        self.linear_model = opt.curve_fit(WeightedLinearFit.linear_fit, self.x, self.y)
        self.m = self.linear_model[0][0]
        self.m_err = np.sqrt(self.linear_model[1][0][0])
        self.has_optimized = True

    def calculate(self, x):
        if self.has_optimized:
            return self.m * x

        raise Exception('Need to optimize model first.')


class MalusFit(object):
    def __init__(self, x, y, alpha = None):
        self.x = x
        self.y = y
        self.alpha = alpha

        self.has_optimized = False

        self.model = None

        self.a = 0
        self.a_err = 0
        self.I0 = 0
        self.I0_err = 0

    @staticmethod
    def malus_fit(theta, a, I0):
        return I0 * np.cos(np.deg2rad(theta) * a) ** 2

    def optimize(self, guess = [0, 0]):
        if self.alpha is not None:
            self.model = opt.curve_fit(MalusFit.malus_fit, self.x, self.y, sigma = self.alpha, p0 = guess)
        else:
            self.model = opt.curve_fit(MalusFit.malus_fit, self.x, self.y, p0 = guess)
        self.a = self.model[0][0]
        self.a_err = np.sqrt(self.model[1][0][0])
        self.I0 = self.model[0][1]
        self.I0_err = np.sqrt(self.model[1][1][1])
        self.has_optimized = True

    def calculate(self, theta):
        if self.has_optimized:
            return self.I0 * np.cos(np.deg2rad(theta) * self.a) ** 2

        raise Exception('Need to optimize model first.')


class Data(object):
    def __init__(self, data, precision_error = 0.05, offset = 0.0):
        self.data = np.array(data) - offset
        self.precision_error = precision_error
        self.average = 0
        self.standard_deviation = 0
        self.standard_error = 0
        self.compute_statistics()

    def compute_statistics(self):
        #self.average = np.mean(self.data, axis = -1)
        self.average = []
        self.standard_deviation = []
        self.standard_error = []
        for i in range(len(self.data)):
            self.average.append(np.mean(self.data[i]))
            self.standard_deviation.append(sample_stdev(self.data[i]))
            self.standard_error.append(standard_error(self.data[i]))

        self.average = np.array(self.average)
        self.standard_deviation = np.array(self.standard_deviation)
        self.standard_error = np.array(self.standard_error)
        #self.standard_deviation = sample_stdev(self.data)
        #self.standard_error = standard_error(self.data)
        for i in range(len(self.standard_error)):
            if self.standard_error[i] == 0:
                self.standard_error[i] = self.precision_error

    def invert_data(self):
        inverted_data = Data(1.0 / self.data, self.precision_error)
        inverted_data.standard_error = -self.standard_error / (self.average * self.average)
        return inverted_data

    def covariance(self, other):
        return covariance(self.data, other.data)


def g(x, a0, a1):
    return a0 * np.cos(np.deg2rad(x) * a1) **2


#Blue LED
relativeAngle = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]) #degrees
current = np.array([903, 852, 706, 509, 305, 98, 7, 65, 254, 498, 717, 845, 895]) #nanoamps

#Green LED
#current = np.array([407, 378, 309, 203, 105, 21, -7, 23, 112, 218, 318, 382, 400]) #nanoamps
#relativeAngle = np.arange(0,15*len(current),15) #degrees

x_error = np.array([3 for i in range(len(relativeAngle))])
y_error = np.array([2 for i in range(len(current))])

total_error = np.hypot(x_error, y_error)

fit = MalusFit(relativeAngle, current, total_error)
fit.optimize(guess = [1, 900])

x = np.linspace(min(relativeAngle), max(relativeAngle), 100)
y = fit.calculate(x)

plt.figure()
plt.title('Green LED: Current vs Polarization Angle')
plt.xlabel('Polarization Angle (deg)')
plt.ylabel('Current (nA)')
plt.text(50, 400, "$I_0$ = %5.0f \u00b1 %5.0f lux" % (fit.I0, fit.I0_err))
plt.text(50, 375, "$a$ = %5.4f \u00b1 %5.4f" % (fit.a, fit.a_err))
plt.text(50, 350, '$I(\\theta)=I_0\\cos^2(a\\cdot\\theta)$')
plt.errorbar(relativeAngle, current, fmt = 'o', xerr = x_error, yerr = y_error, label = 'Data with Error Bars')
plt.plot(x, y, label = 'Best Malus Fit')
plt.legend()

# i = cos^2(pi * theta / 180)
# i_err = pi * theta_err * sin(pi * theta / 90) / 180
intensity = np.cos(np.deg2rad(relativeAngle)) ** 2
intensity_error = math.pi * x_error * np.sin(math.pi * relativeAngle / 90.0) / 180.0

total_error = np.hypot(410.0 * intensity_error, y_error) # 878 comes from initial linear best fit

linear_fit = WeightedLinearFit(intensity, current, alpha = total_error)
linear_fit.optimize(guess = [0, 0])

x = np.linspace(min(intensity), max(intensity), 100)
y = linear_fit.calculate(x)

chi = chi_squared(linear_fit.calculate(intensity), current, total_error)
reduced_chi = chi / (len(intensity) - 2)

plt.figure()
plt.title('Green LED: Current vs Relative Intensity')
plt.xlabel('Relative Intensity $i$')
plt.ylabel('Current $I$ (nA)')
plt.text(0, 300, "$m$ = %5.1f \u00b1 %5.1f nA" % (linear_fit.m, linear_fit.m_err))
plt.text(0, 275, "$c$ = %5.1f \u00b1 %5.1f nA" % (linear_fit.c, linear_fit.c_err))
plt.text(0, 250, '$I(i)=mi+c$')
plt.text(0, 225, "$\\chi$ = %5.2f" % chi)
plt.text(0, 200, "$\\tilde{\\chi}$ = %5.2f" % reduced_chi)
plt.errorbar(intensity, current, fmt = 'o', xerr = intensity_error, yerr = y_error, label = 'Data with Error Bars')
plt.plot(x, y, label = 'Best Weighted Linear Fit')
plt.legend()

plt.show()

