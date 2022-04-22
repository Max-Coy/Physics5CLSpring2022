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


# --------------------- Experiment 1 ---------------------
length_error = 0.05 # cm

can_diameter = 4.5 # cm
can_height = 11.1 # cm
stopper_depth = 0.8 # cm
can_volume = np.pi * (can_diameter / 2.) ** 2 * (can_height - stopper_depth)

can_volume_error = np.sqrt(((np.pi / 2.) * (can_diameter * length_error) * (can_height - stopper_depth)) ** 2 +
                           2. * ((np.pi / 4.) * can_diameter * can_diameter * length_error) ** 2)

print(can_volume, can_volume_error)

tube_diameter = 0.4
tube_length = 63.0
tube_offshoot = 2.5
tube_volume = np.pi * (tube_diameter / 2) ** 2 * (tube_length + tube_offshoot)

tube_volume_error = np.sqrt(((np.pi / 2.) * (tube_diameter * length_error) * (tube_length + tube_offshoot)) ** 2 +
                            2. * ((np.pi / 4.) * tube_diameter * tube_diameter * length_error) ** 2)

print(tube_volume, tube_volume_error)

chamber_diameter = 3.25 # cm
chamber_error = 0.01 # cm

height = np.array([13, 11, 10, 9, 8, 6, 5, 3]) * 0.1 # cm
temperature = np.array([44.5, 39.4, 36.6, 33.6, 29.7, 24.7, 21.8, 17.5]) # C
temperature_error = np.array([0.1] * len(temperature))

chamber_volume = height * np.pi * (chamber_diameter / 2) ** 2
chamber_error = np.sqrt(((np.pi / 4.) * chamber_diameter * chamber_diameter * length_error) ** 2 +
                             ((np.pi / 2.) * (height * chamber_diameter * chamber_error)) ** 2)

print(chamber_volume, chamber_error)

total_volume = can_volume + tube_volume + height * np.pi * (chamber_diameter / 2) ** 2
total_volume_error = np.sqrt(can_volume_error ** 2 + tube_volume_error ** 2 + chamber_error ** 2)

print(total_volume, total_volume_error)

charles_model = WeightedLinearFit(total_volume, temperature, alpha = 3.359 * total_volume_error)
charles_model.optimize()

predicted = charles_model.calculate(total_volume)
chi = chi_squared(predicted, temperature, 3.359 * total_volume_error)
reduced_chi = chi / (len(height) - 2)

x = np.linspace(min(total_volume), max(total_volume), 50)
y = charles_model.calculate(x)

plt.figure()
plt.title('Temperature of Canister vs. Volume of Chamber')
plt.xlabel('Volume of Chamber ($cm^3$)')
plt.ylabel('Temperature of Canister ($^\\circ C$)')
plt.text(170, 40, "$m$ = %5.3f \u00b1 %5.3f $^\\circ C/cm^3$" % (charles_model.m, charles_model.m_err))
plt.text(170, 37, "$c$ = %5.0f \u00b1 %5.0f $^\\circ C$" % (charles_model.c, charles_model.c_err))
plt.text(170, 34, "$T=mV+c$")
plt.text(170, 31, "$\\chi$ = %5.3f" % chi)
plt.text(170, 28, "$\\tilde{\\chi}$ = %5.4f" % reduced_chi)
plt.errorbar(total_volume, temperature, xerr = total_volume_error, yerr = temperature_error, fmt = 'o', label = 'Data with Error Bars')
plt.plot(x, y, label = 'Weighted Linear Fit')
plt.legend()
#plt.show()


# --------------------- Experiment 3 ---------------------
height100 = np.array([69, 67, 64, 61, 58, 56, 53, 50, 47, 44, 41, 38, 35])
times100 = np.arange(0, height100.shape[0] * 30, 30)

height50 = np.array([71, 69, 68, 66, 65, 63, 62, 60, 58, 57, 56, 54, 53])
times50 = np.arange(0, height50.shape[0] * 30, 30)

height0 = np.array([73, 72, 71, 70, 69, 68, 67, 66, 65, 65, 64, 63, 62])
times0 = np.arange(0, height0.shape[0] * 30, 30)

leak100_fit = WeightedLinearFit(times100, height100)
leak100_fit.optimize()

leak50_fit = WeightedLinearFit(times50, height50)
leak50_fit.optimize()

leak0_fit = WeightedLinearFit(times0, height0)
leak0_fit.optimize()

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(leak100_fit.m, leak100_fit.m_err)
print(leak50_fit.m, leak50_fit.m_err)
print(leak0_fit.m, leak0_fit.m_err)

x = np.linspace(min(times0), max(times0), 100)
y100 = leak100_fit.calculate(x)
y50 = leak50_fit.calculate(x)
y0 = leak0_fit.calculate(x)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(times100,height100,label='100 g Data')
ax[0].scatter(times50,height50,label='50 g Data')
ax[0].scatter(times0,height0,label='0 g Data')
ax[0].plot(x, y100, label = '100 g Linear Fit')
ax[0].plot(x, y50, label = '50 g Linear Fit')
ax[0].plot(x, y0, label = '0 g Linear Fit')
ax[0].set_title('Gas Chamber Air Leakage')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Chamber Height (mm)')
ax[0].legend(loc='lower left')
ax[0].set_xlim(0,np.max(np.concatenate([times100,times50])))
ax[0].set_ylim(0,np.max(np.concatenate([height50, height100])))

ax[1].scatter(times100,height100/np.max(height100),label='100 grams')
ax[1].scatter(times50,height50/np.max(height50),label='50 grams')
ax[1].scatter(times0,height0/np.max(height0),label='No Weight')
ax[1].set_title('Normalized Air Leakage')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Normalized Chamber Height')
ax[1].legend(loc='lower left')
ax[1].set_xlim(0, np.max(np.concatenate([times100, times50])))
ax[1].set_ylim(0, 1)

print("------------------ Experiment 3 ------------------")

g = 981 # cm / s^2
K = 273.15

piston_mass = 48.5 # g
piston_mass_error = 0.6 # g

piston_diameter = 3.25 # cm
initial_cold_bath_temperature = 0 # C
initial_hot_bath_temperature = 47.4 # C

piston_area = (np.pi / 4.) * piston_diameter ** 2
piston_area_error = (np.pi / 2.) * piston_diameter * 0.01

temperature50 = np.array([0.1, 47.4, 47.7, 0.0, 0.0, 47.3, 47.3, 0.2, 0.2, 47.5, 47.5, 0.2]) + K
height50 = np.array([30, 56, 57, 26, 26, 47, 47, 16, 14, 40, 41, 10])
mass50 = np.array([50, 50, 0, 0] * 3) + piston_mass
volume50 = height50 * piston_area
pressure50 = mass50 * g * piston_area

height_error = 0.5
volume_error = volume50 * np.sqrt((height_error / height50) ** 2 + (piston_area_error / piston_area) ** 2)
pressure_error = pressure50 * np.sqrt((piston_mass_error / mass50) ** 2 + (piston_area_error / piston_area) ** 2)

gas_work_add_mass = (pressure50[0::4] + pressure50[1::4]) * (volume50[1::4] - volume50[0::4]) / 2.
gas_work_remove_mass = (pressure50[2::4] + pressure50[3::4]) * (volume50[3::4] - volume50[2::4]) / 2.

gas_work_add_mass_error = np.sqrt((pressure_error[0::4] * (volume50[1::4] - volume50[0::4])) ** 2 +
                                  (pressure_error[1::4] * (volume50[1::4] - volume50[0::4])) ** 2 +
                                  (pressure50[0::4] + pressure50[1::4]) * volume_error[1::4] +
                                  (pressure50[0::4] + pressure50[1::4]) * volume_error[0::4]) / 2.

gas_work_remove_mass_error = np.sqrt((pressure_error[2::4] * (volume50[3::4] - volume50[2::4])) ** 2 +
                                  (pressure_error[3::4] * (volume50[3::4] - volume50[2::4])) ** 2 +
                                  (pressure50[2::4] + pressure50[3::4]) * volume_error[3::4] +
                                  (pressure50[2::4] + pressure50[3::4]) * volume_error[2::4]) / 2.

mechanical_work_add_mass = (mass50[0::4] - piston_mass) * g * (height50[3::4] - height50[0::4])
mechanical_work_remove_mass = (mass50[0::4] - piston_mass) * g * (height50[2::4] - height50[1::4])

mechanical_work_add_mass_error = np.sqrt((piston_mass_error * g * (height50[3::4] - height50[0::4])) ** 2 +
                                         (mass50[0::4] - piston_mass) * g * (height_error) ** 2 +
                                         (mass50[0::4] - piston_mass) * g * (height_error) ** 2)

mechanical_work_remove_mass_error = np.sqrt((piston_mass_error * g * (height50[2::4] - height50[1::4])) ** 2 +
                                         (mass50[0::4] - piston_mass) * g * (height_error) ** 2 +
                                         (mass50[0::4] - piston_mass) * g * (height_error) ** 2)

print("~~~~~~~~~~~~ 50 g ~~~~~~~~~~~~")

print(volume50, volume_error)
print(pressure50, pressure_error)

print(gas_work_add_mass, gas_work_add_mass_error)
print(gas_work_remove_mass, gas_work_remove_mass_error)
print(mechanical_work_add_mass, mechanical_work_add_mass_error)
print(mechanical_work_remove_mass, mechanical_work_remove_mass_error)

temperature100 = np.array([0, 47.2, 47.2, 0.2, 0.2, 47.2, 47.3, 0.1, 0.1, 47.2, 47.2, 0.2]) + K
height100 = np.array([33, 58, 60, 30, 26, 53, 53, 22, 19, 44, 46, 16])
mass100 = np.array([100, 100, 0, 0] * 3) + piston_mass
volume100 = height100 * piston_area
pressure100 = mass100 * g * piston_area

height_error = 0.5
volume_error = volume100 * np.sqrt((height_error / height100) ** 2 + (piston_area_error / piston_area) ** 2)
pressure_error = pressure100 * np.sqrt((piston_mass_error / mass100) ** 2 + (piston_area_error / piston_area) ** 2)

gas_work_add_mass = (pressure100[0::4] + pressure100[1::4]) * (volume100[1::4] - volume100[0::4]) / 2.
gas_work_remove_mass = (pressure100[2::4] + pressure100[3::4]) * (volume100[3::4] - volume100[2::4]) / 2.

gas_work_add_mass_error = np.sqrt((pressure_error[0::4] * (volume100[1::4] - volume100[0::4])) ** 2 +
                                  (pressure_error[1::4] * (volume100[1::4] - volume100[0::4])) ** 2 +
                                  (pressure100[0::4] + pressure100[1::4]) * volume_error[1::4] +
                                  (pressure100[0::4] + pressure100[1::4]) * volume_error[0::4]) / 2.

gas_work_remove_mass_error = np.sqrt((pressure_error[2::4] * (volume100[3::4] - volume100[2::4])) ** 2 +
                                  (pressure_error[3::4] * (volume100[3::4] - volume100[2::4])) ** 2 +
                                  (pressure100[2::4] + pressure100[3::4]) * volume_error[3::4] +
                                  (pressure100[2::4] + pressure100[3::4]) * volume_error[2::4]) / 2.

mechanical_work_add_mass = (mass100[0::4] - piston_mass) * g * (height100[3::4] - height100[0::4])
mechanical_work_remove_mass = (mass100[0::4] - piston_mass) * g * (height100[2::4] - height100[1::4])

mechanical_work_add_mass_error = np.sqrt((piston_mass_error * g * (height100[3::4] - height100[0::4])) ** 2 +
                                         (mass100[0::4] - piston_mass) * g * (height_error) ** 2 +
                                         (mass100[0::4] - piston_mass) * g * (height_error) ** 2)

mechanical_work_remove_mass_error = np.sqrt((piston_mass_error * g * (height100[2::4] - height100[1::4])) ** 2 +
                                         (mass100[0::4] - piston_mass) * g * (height_error) ** 2 +
                                         (mass100[0::4] - piston_mass) * g * (height_error) ** 2)

print("~~~~~~~~~~~~ 100 g ~~~~~~~~~~~~")

print(volume100, volume_error)
print(pressure100, pressure_error)

print(gas_work_add_mass, gas_work_add_mass_error)
print(gas_work_remove_mass, gas_work_remove_mass_error)
print(mechanical_work_add_mass, mechanical_work_add_mass_error)
print(mechanical_work_remove_mass, mechanical_work_remove_mass_error)

plt.show()

