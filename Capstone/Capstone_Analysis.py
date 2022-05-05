import numpy as np
import scipy.optimize as fitter
import matplotlib.pyplot as plt


def parse_resistivity_file(filename):
    data = np.loadtxt(filename, dtype = str, delimiter = ' ')
    temperature = data[:, 1].astype(float)
    resistivity = data[:, 2]
    for i in range(resistivity.shape[0]):
        resistivity[i] = resistivity[i].strip().replace(',', '.')

    resistivity = np.array(resistivity, dtype = 'float')
    return resistivity, temperature


class QuadraticFit(object):
    def __init__(self, x, y, alpha = None):
        self.x = x
        self.y = y
        self.alpha = alpha

        self.model = None
        self.a0 = 0
        self.a0_err = 0
        self.a1 = 0
        self.a1_err = 0
        self.a2 = 0
        self.a2_err = 0

    @staticmethod
    def fit(x, a0, a1, a2):
        return a0 + a1 * x + a2 * x * x

    def optimize(self, guess = [0, 0, 0]):
        self.model = fitter.curve_fit(QuadraticFit.fit, self.x, self.y, p0 = guess, sigma = self.alpha)
        self.a0 = self.model[0][0]
        self.a1 = self.model[0][1]
        self.a2 = self.model[0][2]
        self.a0_err = np.sqrt(self.model[1][0][0])
        self.a1_err = np.sqrt(self.model[1][1][1])
        self.a2_err = np.sqrt(self.model[1][2][2])

    def calculate(self, x):
        return self.a0 + self.a1 * x + self.a2 * x * x


class SquareRootFit(object):
    def __init__(self, x, y, alpha = None):
        self.x = x
        self.y = y
        self.alpha = alpha

        self.model = None
        self.a0 = 0
        self.a0_err = 0
        self.a1 = 0
        self.a1_err = 0
        self.a2 = 0
        self.a2_err = 0

    @staticmethod
    def fit(x, a0, a1, a2):
        return a0 + np.sqrt(a1 * x + a2)

    def optimize(self, guess = [0, 0, 0]):
        self.model = fitter.curve_fit(SquareRootFit.fit, self.x, self.y, p0 = guess, sigma = self.alpha,
                                      bounds = ([-np.inf, 1000, 0], [np.inf, np.inf, np.inf]))
        self.a0 = self.model[0][0]
        self.a1 = self.model[0][1]
        self.a2 = self.model[0][2]
        self.a0_err = np.sqrt(self.model[1][0][0])
        self.a1_err = np.sqrt(self.model[1][1][1])
        self.a2_err = np.sqrt(self.model[1][2][2])

    def calculate(self, x):
        return self.a0 + np.sqrt(self.a1 * x + self.a2)


class ExponentialFit(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.model = None
        self.a0 = 0
        self.a0_err = 0
        self.a1 = 0
        self.a1_err = 0
        self.a2 = 0
        self.a2_err = 0

    @staticmethod
    def fit(x, a0, a1, a2):
        return a0 + a1 * np.exp(-a2 * x)

    def optimize(self, guess = [0, 0, 0]):
        self.model = fitter.curve_fit(ExponentialFit.fit, self.x, self.y, p0 = guess)
        self.a0 = self.model[0][0]
        self.a1 = self.model[0][1]
        self.a2 = self.model[0][2]
        self.a0_err = np.sqrt(self.model[1][0][0])
        self.a1_err = np.sqrt(self.model[1][1][1])
        self.a2_err = np.sqrt(self.model[1][2][2])

    def calculate(self, x):
        return self.a0 + self.a1 * np.exp(-self.a2 * x)


K = 273.15
resistivity, temperature = parse_resistivity_file('tungsten_resistivity.txt')
resistivity_error = np.array([0.01 for i in range(len(resistivity))]) # microOhm * cm
temperature_error = np.array([1 for i in range(len(temperature))])

resistivity_fit = QuadraticFit(temperature, resistivity, alpha = temperature_error)
resistivity_fit.optimize()
x = np.linspace(min(temperature), max(temperature), 100)
y = resistivity_fit.calculate(x)

# plt.figure()
# plt.title('Resistivity of Tungsten vs. Temperature')
# plt.ylabel(r'Resistivity ($\mu\Omega\cdot cm$)')
# plt.xlabel('Temperature (K)')
# plt.text(250, 100, "$a_0$ = %5.2f \u00b1 %5.2f $\\mu\\Omega\\cdot cm$" % (resistivity_fit.a0, resistivity_fit.a0_err))
# plt.text(250, 90, "$a_1$ = %5.5f \u00b1 %5.5f $\\frac{\\mu\\Omega\\cdot cm}{K}$" % (resistivity_fit.a1, resistivity_fit.a1_err))
# plt.text(250, 80, "$a_2$ = %5.9f \u00b1 %5.9f $\\frac{\\mu\\Omega\\cdot cm}{K^2}$" % (resistivity_fit.a2, resistivity_fit.a2_err))
# plt.text(250, 70, "$\\rho(T)=a_0+a_1T+a_2T^2$")
# plt.errorbar(temperature, resistivity, xerr = temperature_error, yerr = resistivity_error, fmt = 'o', label = 'Data with Error Bars')
# plt.plot(x, y, label = 'Quadratic Fit')
# plt.legend()


temperature_fit = QuadraticFit(resistivity, temperature, alpha = resistivity_error)
temperature_fit.optimize(guess = [0, 1000, 0])
x = np.linspace(min(resistivity), max(resistivity), 100)
y = temperature_fit.calculate(x)

# plt.figure()
# plt.title('Temperature of Tungsten vs. Resistivity')
# plt.ylabel('Temperature (K)')
# plt.xlabel(r'Resistivity ($\mu\Omega\cdot cm$)')
# plt.text(0, 3000, "$a_0$ = %5.1f \u00b1 %5.1f K" % (temperature_fit.a0, temperature_fit.a0_err))
# plt.text(0, 2700, "$a_1$ = %5.2f \u00b1 %5.2f $\\frac{K}{\\mu\\Omega\\cdot cm}$" % (temperature_fit.a1, temperature_fit.a1_err))
# plt.text(0, 2400, "$a_2$ = %5.4f \u00b1 %5.4f $\\frac{K}{\\mu\\Omega^2\\cdot cm^2}$" % (temperature_fit.a2, temperature_fit.a2_err))
# plt.text(0, 2100, "$T(\\rho)=a_0+a_1\\rho+a_2\\rho^2$")
# plt.errorbar(resistivity, temperature, xerr = resistivity_error, yerr = temperature_error, fmt = 'o', label = 'Data with Error Bars')
# plt.plot(x, y, label = 'Square Root Fit')
# plt.legend()


expansion_data = np.array([[20 + K, 1000 + K, 1800 + K], [4.5e-6, 6e-6, 7e-6]])
temperature_expansion_data = expansion_data[0]
expansion_coefficient_data = expansion_data[1]

expansion_fit = ExponentialFit(temperature_expansion_data, expansion_coefficient_data)
expansion_fit.optimize(guess = [9.2, 4.7, 0.8/2000])
x = np.linspace(0, 2500, 100)
y = expansion_fit.calculate(x)

# plt.figure()
# plt.title('Thermal Expansion of Tungsten vs. Temperature')
# plt.ylabel(r'Thermal Expansion Coefficient ($K^{-1}$)')
# plt.xlabel('Temperature (K)')
# plt.text(0, 6.5e-6, "$a_0$ = %5.1f \u00b1 %5.1f K" % (expansion_fit.a0, expansion_fit.a0_err))
# plt.text(0, 6.2e-6, "$a_1$ = %5.2f \u00b1 %5.2f $\\frac{K}{\\mu\\Omega\\cdot cm}$" % (expansion_fit.a1, expansion_fit.a1_err))
# plt.text(0, 5.9e-6, "$a_2$ = %5.4f \u00b1 %5.4f $\\frac{K}{\\mu\\Omega^2\\cdot cm^2}$" % (expansion_fit.a2, expansion_fit.a2_err))
# plt.text(0, 5.6e-6, "$T(\\rho)=a_0+a_1\\rho+a_2\\rho^2$")
# plt.plot(temperature_expansion_data, expansion_coefficient_data, 'o', label = 'Data')
# plt.plot(x, y, label = 'Exponential Fit')
# plt.legend()


length = 6.0998179802751835 # cm
length_erorr = 0.0809 # cm
cross_sectional_area = 0.000005856 # cm^2
cross_sectional_area_error = 0.000000017 # cm^2

temperature_list = np.linspace(min(temperature), max(temperature), 100)
resistivity_list = resistivity_fit.calculate(temperature_list) / 1e6

resistance_no_expansion = resistivity_list * length / cross_sectional_area
resistance_expansion = resistivity_list * length / (cross_sectional_area * (1. + expansion_fit.calculate(temperature_list)))

# plt.figure()
# plt.title('Tungsten Resistance vs. Temperature')
# plt.ylabel(r'Tungsten Resistance ($\Omega$)')
# plt.xlabel('Temperature (K)')
# plt.plot(temperature_list, resistance_no_expansion, label = 'No Thermal Expansion')
# plt.plot(temperature_list, resistance_expansion, label = 'With Thermal Expansion')
# plt.legend()

# plt.figure()
# plt.title('Tungsten Resistance Residuals vs. Temperature')
# plt.ylabel(r'Tungsten Resistance Residuals ($\Omega$)')
# plt.xlabel('Temperature (K)')
# plt.plot(temperature_list, resistance_no_expansion - resistance_expansion, label = 'Thermal Expansion Difference')


def resistance_to_temperature(resistance, length, cross_sectional_area, resistance_error = 0, length_error = 0, cross_sectional_area_error = 0):
    resistivity = resistance * 1e6 * cross_sectional_area / length
    resistivity_error = resistivity * np.sqrt((resistance_error / resistance) ** 2 + (length_error / length) ** 2 +
                                              (cross_sectional_area_error / cross_sectional_area) ** 2)
    temperature_fit_error = np.sqrt(temperature_fit.a0_err ** 2 + (resistivity * temperature_fit.a1_err) ** 2 +
                                    (temperature_fit.a1 * resistivity_error) ** 2 +
                                    (resistivity * resistivity * temperature_fit.a2_err) ** 2 +
                                    4. * (temperature_fit.a2 * resistivity * resistivity_error) ** 2)
    return temperature_fit.calculate(resistivity), temperature_fit_error


print(resistance_to_temperature(5.4, length, cross_sectional_area, 0.1, length_erorr, cross_sectional_area_error))


# plt.show()

