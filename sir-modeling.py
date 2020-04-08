from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

S_0 = 13500
I_0 = 2
R_0 = 0

class Learner(object):
	def __init__(self, country, loss):
		self.country = country
		self.loss = loss

	def load_confirmed(self, country):
		"""
			Load confirmed cases downloaded from HDX
		"""
		df = pd.read_csv('data/time_series_19-covid-Confirmed.csv')
		country_df = df[df['Country_Region'] == country]
		return country_df.iloc[0].loc['1/22/20':]

	def load_recovered(self, country):
		"""
			Load recovered cases downloaded from HDX
		"""
		df = pd.read_csv('data/time_series_19-covid-Recovered.csv')
		country_df = df[df['Country/Region'] == country]
		return country_df.iloc[0].loc['1/22/20':]

	def extend_index(self, index, new_size):
		values = index.values
		current = datetime.strptime(index[-1], '%m/%d/%y')
		while len(values) < new_size:
			current = current + timedelta(days=1)
			values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
		return values

	def predict(self, beta, gamma, data):
		"""
			Predict how the number of people in each compartment can be changed through time toward the future.
			The model is formulated with the given beta and gamma.
		"""
		predict_range = 150
		new_index = self.extend_index(data.index, predict_range)
		size = len(new_index)
		def SIR(t, y):
			S = y[0]
			I = y[1]
			R = y[2]
			return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
		extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
		return new_index, extended_actual, solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1))

	def train(self):
		"""
			Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
		"""
		confirmed_data = self.load_confirmed(self.country)
		recovered_data = self.load_recovered(self.country)
		optimal = minimize(
			loss,
			[0.001, 0.001],
			args=(confirmed_data, recovered_data),
			method='L-BFGS-B',
			bounds=[(0.00000001, 0.4), (0.00000001, 0.4)]
		)
		beta, gamma = optimal.x
		print(f'Beta: {beta}, Gamma: {gamma}, R0: {beta/gamma}')
		new_index, extended_actual, prediction = self.predict(beta, gamma, confirmed_data)
		df = pd.DataFrame({
			'Actual': extended_actual,
			'S': prediction.y[0],
			'I': prediction.y[1],
			'R': prediction.y[2]
		}, index=new_index)
		fig, ax = plt.subplots(figsize=(15, 10))
		ax.set_title(self.country)
		df.plot(ax=ax)
		fig.savefig(f"{self.country}.png")

# def loss(point, data):
    # """
    # RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
    # """
    # size = len(data)
    # beta, gamma = point
    # def SIR(t, y):
    #     S = y[0]
    #     I = y[1]
    #     R = y[2]
    #     return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
    # solution = solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)
    # return np.sqrt(np.mean((solution.y[1] - data)**2))

def loss(point, data, recovered):
    size = len(data)
    beta, gamma = point
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
    solution = solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] - data)**2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered)**2))
    # Put more emphasis on recovered people
    alpha = 0.1
    return alpha * l1 + (1 - alpha) * l2

my_learner = Learner('US', loss)
my_learner.train()
