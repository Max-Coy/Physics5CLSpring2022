import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import math
import csv


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


class HeatConduction(object):
    def __init__(self, Tbath, T0, L, kappa, rho, c, N):
        self.alpha = kappa / (rho * c)
        self.Tbath = Tbath
        self.T0 = T0
        self.L = L
        self.N = N

    def summation_term(self, n, t):
        sign = 1.0 if n % 2 == 0 else -1.0
        k = math.pi * (n + 1.0 / 2.0) / self.L
        return 4.0 * sign * (math.exp(-self.alpha * k * k * t)) / (math.pi * (2.0 * n + 1.0))

    def endpoint_temperature(self, t):
        temperature = self.Tbath
        for i in range(self.N):
            temperature += (self.T0 - self.Tbath) * self.summation_term(i, t)

        return temperature


class HeatConductionFit(object):
    def __init__(self, t, T, Tbath, T0, L, rho, c, N):
        self.t = t
        self.T = T

        self.Tbath = Tbath
        self.T0 = T0
        self.L = L
        self.rho = rho
        self.c = c
        self.N = N

        self.kappa = 0.07
        self.kappa_err = 0
        self.t0 = 0
        self.t0_err = 0

        self.model = None

    def summation_term(self, alpha, n, t, t0):
        sign = 1.0 if n % 2 == 0 else -1.0
        k = math.pi * (n + 1.0 / 2.0) / self.L
        return 4.0 * sign * (np.exp(-alpha * k * k * (t - t0))) / (math.pi * (2.0 * n + 1.0))

    def endpoint_temperature(self, t, t0, kappa):
        temperature = self.Tbath
        for i in range(self.N):
            temperature += (self.T0 - self.Tbath) * self.summation_term(kappa / (self.rho * self.c), i, t, t0)

        return temperature

    def optimize(self, guess = [0]):
        self.model = opt.curve_fit(self.endpoint_temperature, self.t, self.T, p0 = guess,
                                   bounds = ([0, 0.1]))
        self.kappa = self.model[0][0]
        #self.t0 = self.model[0][1]
        self.kappa_err = np.sqrt(self.model[1][0][0])
        #self.t0_err = np.sqrt(self.model[1][1][1])

    def calculate(self, t):
        return self.endpoint_temperature(t, self.t0, self.kappa / (self.rho * self.c))

    def compute_cost(self, kappa, t0):
        return np.sqrt(np.mean(np.power(self.T - self.endpoint_temperature(self.t, t0, kappa / (self.rho * self.c)), 2)))


def read_data(filename):
    K = 273.15
    times = []
    temperatures = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        next(csv_reader)
        for row in csv_reader:
            times.append(float(row[0]))
            temperatures.append(float(row[1]) + K)

    return np.array([times, temperatures])


# --------------------- Experiment 1 ---------------------
K = 273.15
hot_mass = np.array([207, 146, 217, 211]) # grams
hot_temperature = np.array([47.5, 47.2, 47.6, 47.4]) # Kelvin
cold_mass = np.array([129, 209, 172, 193])
cold_temperature = np.array([17.5, 17.4, 17.4, 17.4])
equilibrium_temperature = np.array([35.7, 29.7, 34.2, 33.0])

mass_precision_error = 1.0 # g
temperature_precision_error = 0.1 # g

numerator = hot_mass * hot_temperature + cold_mass * cold_temperature
denominator = cold_mass + hot_mass
equilibrium_temperature_theory = numerator / denominator
equilibrium_temperature_error = np.sqrt((mass_precision_error * hot_temperature / denominator) ** 2 +
                                        (mass_precision_error * cold_temperature / denominator) ** 2 +
                                        2.0 * (mass_precision_error * numerator / (denominator ** 2)) ** 2 +
                                        (temperature_precision_error * hot_mass / denominator) ** 2 +
                                        (temperature_precision_error * cold_mass / denominator) ** 2)

chi = chi_squared(equilibrium_temperature_theory, equilibrium_temperature, equilibrium_temperature_error)
reduced_chi = chi / 3.0

print(equilibrium_temperature)
print(equilibrium_temperature_theory)
print(equilibrium_temperature_error)
print(chi, reduced_chi)


# --------------------- Experiment 4 ---------------------
K = 273.15
can_height = 7.1 # cm

# Brass
brass_density = 8.73 # g / cm^3
brass_specific_heat = 0.38 # J / g * K

brass_depth = 4.0 # cm
initial_cold_bath_temperature_brass = 0.7 + K # K
initial_brass_temperature = 27.1 + K # K
probe_depth = 3.4 # cm
brass_length = 20.5 # cm
brass_effective_length = brass_length - probe_depth - brass_depth

# Copper
copper_density = 8.96 # g / cm^3
copper_specific_heat = 0.385 # J / g * K

copper_depth = 3.5 # cm
initial_cold_bath_temperature_copper = 0.3 + K
initial_copper_temperature = 24.0 + K
probe_depth = 3.4 # cm
copper_length = 20.5 # cm
copper_effective_length = copper_length - probe_depth - copper_depth

# Aluminum
aluminum_thermal_conductivity = 2.05 # W / cm * K
aluminum_density = 2.7 # g / cm^3
aluminum_specific_heat = 0.89 # J / g * K

initial_aluminum_temperature = 25.5 + K
initial_cold_bath_temperature_aluminum = 0.6 + K
aluminum_depth = 4.5 # cm
probe_depth = 3.4 # cm
aluminum_length = 20.5 # cm
aluminum_effective_length = aluminum_length - aluminum_length - probe_depth

brass_data = read_data('brass_rod.csv')
copper_data = read_data('copper_rod.csv')
aluminum_data = read_data('aluminum_rod.csv')
initial_brass_temperature = brass_data[1][0]
initial_copper_temperature = copper_data[1][0]
initial_aluminum_temperature = aluminum_data[1][0]

brass_fit = HeatConductionFit(brass_data[0], brass_data[1], initial_cold_bath_temperature_brass, initial_brass_temperature,
                              brass_effective_length, brass_density, brass_specific_heat, 100)
copper_fit = HeatConductionFit(copper_data[0], copper_data[1], initial_cold_bath_temperature_copper, initial_copper_temperature,
                               copper_effective_length, copper_density, copper_specific_heat, 100)
aluminum_fit = HeatConductionFit(aluminum_data[0], aluminum_data[1], initial_cold_bath_temperature_aluminum, initial_aluminum_temperature,
                                 aluminum_effective_length, aluminum_density, aluminum_specific_heat, 100)

n = 1000
max_val = 3
m = 1
max_t0 = 0
kappas = []
t0s = []
brass_costs = []
copper_costs = []
aluminum_costs = []
for i in range(n):
    for j in range(m):
        kappa = max_val * i / n
        t0 = max_t0 * j / m
        kappas.append(kappa)
        t0s.append(t0)
        brass_costs.append(brass_fit.compute_cost(kappa, t0))
        copper_costs.append(copper_fit.compute_cost(kappa, t0))
        aluminum_costs.append(aluminum_fit.compute_cost(kappa, t0))

brass_kappa_optimal = kappas[brass_costs.index(min(brass_costs))]
#brass_t0_optimal = t0s[brass_costs.index(min(brass_costs))]
print(brass_kappa_optimal, min(brass_costs) / len(brass_data[0]))
#print(brass_t0_optimal, min(brass_costs) / len(brass_data[0]))

copper_kappa_optimal = kappas[copper_costs.index(min(copper_costs))]
print(copper_kappa_optimal, min(copper_costs) / len(copper_data[0]))

aluminum_kappa_optimal = kappas[aluminum_costs.index(min(aluminum_costs))]
print(aluminum_kappa_optimal, min(aluminum_costs) / len(aluminum_data[0]))

brass_fit.kappa = brass_kappa_optimal
T_brass = brass_fit.calculate(brass_data[0])

copper_fit.kappa = copper_kappa_optimal
T_copper = copper_fit.calculate(copper_data[0])

aluminum_fit.kappa = aluminum_kappa_optimal
T_aluminum = aluminum_fit.calculate(aluminum_data[0])

plt.figure()
plt.title('Temperature RMS Error vs. Thermal Conductivity of Brass')
plt.xlabel('Thermal Conductivity of Brass ($\\kappa$)')
plt.ylabel('Temperature RMS Error (K)')
plt.plot(kappas, brass_costs)

plt.figure()
plt.title('Temperature RMS Error vs. Thermal Conductivity of Copper')
plt.xlabel('Thermal Conductivity of Copper ($\\kappa$)')
plt.ylabel('Temperature RMS Error (K)')
plt.plot(kappas, copper_costs)

plt.figure()
plt.title('Temperature RMS Error vs. Thermal Conductivity of Aluminum')
plt.xlabel('Thermal Conductivity of Aluminum ($\\kappa$)')
plt.ylabel('Temperature RMS Error (K)')
plt.plot(kappas, aluminum_costs)

plt.figure()
plt.title('Brass Rod Heat Conduction Temperature vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.text(0, 287.5, "$\\kappa$ = %5.4f \u00b1 %5.4f $\\frac{W}{cm\\cdot K}$" % (brass_kappa_optimal, min(brass_costs) / len(brass_data[0])))
plt.plot(brass_data[0], T_brass, label = 'Heat Conduction Model')
plt.plot(brass_data[0], brass_data[1], label = 'Experimental Data')
plt.legend()

plt.figure()
plt.title('Copper Rod Heat Conduction Temperature vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.text(0, 287, "$\\kappa$ = %5.4f \u00b1 %5.4f $\\frac{W}{cm\\cdot K}$" % (copper_kappa_optimal, min(copper_costs) / len(copper_data[0])))
plt.plot(copper_data[0], T_copper, label = 'Heat Conduction Model')
plt.plot(copper_data[0], copper_data[1], label = 'Experimental Data')
plt.legend()

plt.figure()
plt.title('Aluminum Rod Heat Conduction Temperature vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.text(0, 285, "$\\kappa$ = %5.4f \u00b1 %5.4f $\\frac{W}{cm\\cdot K}$" % (aluminum_kappa_optimal, min(aluminum_costs) / len(aluminum_data[0])))
plt.plot(aluminum_data[0], T_aluminum, label = 'Heat Conduction Model')
plt.plot(aluminum_data[0], aluminum_data[1], label = 'Experimental Data')
plt.legend()
plt.show()

