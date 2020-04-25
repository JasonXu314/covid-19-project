from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import argparse
import os

# TODO - Parse arguments for different model options
parser = argparse.ArgumentParser()

parser.add_argument('--mode', '-m', dest = 'mode', help = 'change the mode of the model (SIR, Linear, ESIR, SEIR); default: SIR', default = 'SIR', choices = ['SIR', 'Linear', 'ESIR', 'SEIR'])
parser.add_argument('--data', '-d', dest = 'include_data', help = 'change the type of data to present in the graph (Actual, S, I, R, E); default: Actual S I R', nargs = '*', default = ['Actual', 'S', 'I', 'R'], choices = ['Actual', 'S', 'I', 'R', 'E'])
parser.add_argument('--folder', '-f', dest = 'folder', default = 'data', help = 'the folder in which to find the data files; defaults to looking in the data folder')
parser.add_argument('--disease', '-D', dest = 'disease', default = 'COVID-19', help = 'the disease to model; defaults to COVID-19')
parser.add_argument('--out', '-o', dest = 'out', default = None, help = 'the name of the graph and csv files; defaults to the name of the disease')
parser.add_argument('--start', '-s', dest = 'start', default = '1/22/20', help = 'the date where the data starts (defaults to the start date of COVID-19 (1/22/20))')
parser.add_argument('--end', '-e', dest = 'end', default = None, help = 'the date where the data stops (defaults to whereever the input data ends)')
parser.add_argument('--incubation', '-i', dest = 'incubation_period', default = None, help = 'the incubation period of the disease (only applicable if using SIRE model; ignored otherwise); none by default')
parser.add_argument('--predict', '-p', dest = 'prediction_range', default = None, help = 'the number of days to predict the course of the disease (defaults to None, meaning the model will not predict beyond the given data)')
args = parser.parse_args()

S_0 = 13500/13501
I_0 = 1/13501
R_0 = 0 # Both are equal to 0/13501
E_0 = 0 # Both are equal to 0/13501

# Running a model for 3.27  million population is quite hard, so here we've reduced the population to 13.5 thousand people, and modified
# the actual stats to match
correction_factor = 13502/3270000 if args.disease == 'COVID-19' else 1

class Learner(object):
	def __init__(self, country):
		self.country = country

	def load_confirmed(self, country):
		"""
			Load confirmed cases
		"""
		df = pd.read_csv(f'{args.folder}/{args.disease}-Confirmed.csv')
		country_df = df[df['Country/Region'] == country]

		if args.end != None:
			confirmed_sums = np.sum([reg.loc[args.start:args.end].values for reg in country_df.iloc], axis = 0)
		else:
			confirmed_sums = np.sum([reg.loc[args.start:].values for reg in country_df.iloc], axis = 0)
		
		if args.end != None:
			new_data = pd.DataFrame(confirmed_sums, country_df.iloc[0].loc[args.start:args.end].index.tolist())
		else:
			new_data = pd.DataFrame(confirmed_sums, country_df.iloc[0].loc[args.start:].index.tolist())
		
		return new_data


	def load_recovered(self, country):
		"""
			Load recovered cases
		"""
		df = pd.read_csv(f'{args.folder}/{args.disease}-Recovered.csv')
		country_df = df[df['Country/Region'] == country]

		if args.end != None:
			out = country_df.iloc[0].loc[args.start:args.end]
		else:
			out = country_df.iloc[0].loc[args.start:]
		
		return out

	def load_exposed(self, country):
		"""
			Load data for exposed persons 
		"""
		df = pd.read_csv(f'{args.folder}/{args.disease}-Exposed.csv')
		country_df = df[df['Country/Region'] == country]

		if args.end != None:
			out = country_df.iloc[0].loc[args.start:args.end]
		else:
			out = country_df.iloc[0].loc[args.start:]
		
		return out


	def extend_index(self, index, new_size):
		values = index.values
		current = datetime.strptime(index[-1], '%m/%d/%y')
		while len(values) < new_size:
			current = current + timedelta(days=1)
			values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
		return values

	def predict(self, data, beta = None, gamma = None, mu = None, sigma = None):
		"""
			Predict how the number of people in each compartment can be changed through time toward the future.
			The model is formulated with the given beta and gamma (or others).
			Returns the "solved" system of initial value problems, to be "graded" by the loss function
		"""
		new_index = self.extend_index(data.index, args.prediction_range if args.prediction_range != None else len(data.index))
		size = len(new_index)
		def model(t, y):
			S = y[0]
			I = y[1]
			R = y[2]

			if args.mode == 'SEIR':
				E = y[3]

			if args.mode == 'Linear':
				return [-beta * S, beta * S - gamma * I, gamma * I]
			elif args.mode == 'SIR':
				return [-beta * S * I, beta * S * I - gamma * I, gamma * I]
			elif args.mode == 'ESIR':
				if mu != None:
					return [mu - beta * S * I - mu * S, beta * S * I - gamma * I - mu * I, gamma * I - mu * R]
				else:
					raise Exception('Expected mu for ESIR model')
			elif args.mode == 'SEIR':
				if mu != None and sigma != None:
					return [mu - beta * S * I - mu * S, beta * S * I - sigma * E - mu * E, sigma * E - gamma * I - mu * I, gamma * I - mu * R]
				elif mu == None:
					raise Exception('Expected mu for SEIR model')
				elif sigma == None:
					raise Exception('Expected sigma for SEIR model')
		
		extended_actual = np.concatenate((data.values.flatten(), [0] * (size - len(data.values))))

		if args.mode == 'SEIR':
			result = solve_ivp(model, [0, size], [S_0,I_0,R_0, E_0], t_eval=np.arange(0, size, 1))
		else:
			result = solve_ivp(model, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)

		return new_index, extended_actual, result

	def train(self):
		"""
			Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
		"""
		confirmed_data = self.load_confirmed(self.country)
		recovered_data = self.load_recovered(self.country)

		if not os.path.isdir('out'):
			os.mkdir('out')

		if args.mode == 'Linear':
			optimal = minimize(
				loss_linear,
				[0.001, 0.001],
				args=(confirmed_data, recovered_data),
				method='L-BFGS-B',
				bounds=[(0.00000001, 0.4), (0.00000001, 0.4)]
			)
			beta, gamma = optimal.x
			print(f'Beta: {beta}, Gamma: {gamma}, R0: {beta/gamma}')
			new_index, extended_actual, prediction = self.predict(confirmed_data, beta = beta, gamma = gamma)
			print(f'Predicted I: {prediction.y[1][-1] * 13500}, Actual I: {extended_actual[-1] * correction_factor}')
			df = compose_df(prediction, extended_actual, correction_factor, new_index)
			with open(f'out/{args.disease}-data.csv', 'w+') as file:
				file.write(f'Beta: {beta}\nGamma: {gamma}\nR0: {beta/gamma}')
		elif args.mode == 'SIR':
			optimal = minimize(
				loss_sir,
				[0.001, 0.001],
				args=(confirmed_data, recovered_data),
				method='L-BFGS-B',
				bounds=[(0.00000001, 0.4), (0.00000001, 0.4)]
			)
			beta, gamma = optimal.x
			print(f'Beta: {beta}, Gamma: {gamma}, R0: {beta/gamma}')
			new_index, extended_actual, prediction = self.predict(confirmed_data, beta = beta, gamma = gamma)
			print(f'Predicted I: {prediction.y[1][-1] * 13500}, Actual I: {extended_actual[-1] * correction_factor}')
			df = compose_df(prediction, extended_actual, correction_factor, new_index)
			with open(f'out/{args.disease}-data.csv', 'w+') as file:
				file.write(f'Beta: {beta}\nGamma: {gamma}\nR0: {beta/gamma}')
		elif args.mode == 'ESIR':
			optimal = minimize(
				loss_esir,
				[0.001, 0.001, 0.001],
				args=(confirmed_data, recovered_data),
				method='L-BFGS-B',
				bounds=[(0.00000001, 0.4), (0.00000001, 0.4), (0.00000001, 0.4)]
			)
			beta, gamma, mu = optimal.x
			print(f'Beta: {beta}, Gamma: {gamma}, Mu: {mu} R0: {beta/(gamma + mu)}')
			new_index, extended_actual, prediction = self.predict(confirmed_data, beta = beta, gamma = gamma, mu = mu)
			print(f'Predicted I: {prediction.y[1][-1] * 13500}, Actual I: {extended_actual[-1] * correction_factor}')
			df = compose_df(prediction, extended_actual, correction_factor, new_index)
			with open(f'out/{args.disease}-data.csv', 'w+') as file:
				file.write(f'Beta: {beta}\nGamma: {gamma}\nMu: {mu}\nR0: {beta/(gamma + mu)}')
		elif args.mode == 'SEIR':
			exposed_data = self.load_exposed(self.country)

			optimal = minimize(
				loss_seir,
				[0.001, 0.001],
				args=(confirmed_data, recovered_data, exposed_data),
				method='L-BFGS-B',
				bounds=[(0.00000001, 0.4), (0.00000001, 0.4), (0.00000001, 0.4), (0.00000001, 0.4)]
			)
			beta, gamma, mu, sigma = optimal.x
			print(f'Beta: {beta}, Gamma: {gamma}, Mu: {mu}, Sigma: {sigma} R0: {(beta * sigma)/((mu + gamma) * (mu + sigma))}')
			new_index, extended_actual, prediction = self.predict(confirmed_data, beta = beta, gamma = gamma, mu = mu)
			print(f'Predicted I: {prediction.y[1][-1] * 13500}, Actual I: {extended_actual[-1] * correction_factor}')
			df = compose_df(prediction, extended_actual, correction_factor, new_index)
			with open(f'out/{args.disease}-data.csv', 'w+') as file:
				file.write(f'Beta: {beta}\nGamma: {gamma}\nMu: {mu}\nSigma: {sigma}\nR0: {(beta * sigma)/((mu + gamma) * (mu + sigma))}')
		fig, ax = plt.subplots(figsize=(15, 10))
		ax.set_title(f'{args.disease} cases over time ({args.mode} Model)')
		df.plot(ax=ax)
		fig.savefig(f"{args.out if args.out != None else args.disease}.png")
		df.to_csv(f'out/{args.disease}-prediction.csv')

def filter_zeroes(arr):
	out = np.array(arr)
	for index in range(len(out)):
		if out[index] == 0:
			out[index] = None
	return out

def compose_df(prediction, actual, correction_factor, index):
	df_dict = {}

	for data in args.include_data:
		if data == 'Actual':
			df_dict['Actual'] = filter_zeroes(actual * correction_factor)
		elif data == 'S':
			df_dict['S'] = prediction.y[0] * 13500
		elif data == 'I':
			df_dict['I'] = prediction.y[1] * 13500
		elif data == 'R':
			df_dict['R'] = prediction.y[2] * 13500
		elif data == 'E':
			df_dict['E'] = prediction.y[3] * 13500

	return pd.DataFrame(df_dict, index=index)

# Loss Functions - used to "train" the model

def loss_linear(point, confirmed, recovered):
	size = len(confirmed)
	beta, gamma = point
	def model(t, y):
		S = y[0]
		I = y[1]
		R = y[2]
		return [-beta * S, beta * S - gamma * I, gamma * I]
	solution = solve_ivp(model, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)
	sol_inf = np.sqrt(np.mean((solution.y[1] - (confirmed.values.flatten() * correction_factor/13500))**2))
	sol_rec = np.sqrt(np.mean((solution.y[2] - (recovered.values * correction_factor/13500))**2))
	return sol_inf * 0.5 + sol_rec * 0.5

def loss_sir(point, confirmed, recovered):
	size = len(confirmed)
	beta, gamma = point
	def model(t, y):
		S = y[0]
		I = y[1]
		R = y[2]
		return [-beta * S * I, beta * S * I - gamma * I, gamma * I]
	solution = solve_ivp(model, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)
	sol_inf = np.sqrt(np.mean((solution.y[1] - (confirmed.values.flatten() * correction_factor/13500))**2))
	sol_rec = np.sqrt(np.mean((solution.y[2] - (recovered.values * correction_factor/13500))**2))
	return sol_inf * 0.5 + sol_rec * 0.5

def loss_esir(point, confirmed, recovered):
	size = len(confirmed)
	beta, gamma, mu = point
	def model(t, y):
		S = y[0]
		I = y[1]
		R = y[2]
		return [mu - beta * S * I - mu * S, beta * S * I - gamma * I - mu * I, gamma * I - mu * R]
	solution = solve_ivp(model, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)
	sol_inf = np.sqrt(np.mean((solution.y[1] - (confirmed.values.flatten() * correction_factor/13500))**2))
	sol_rec = np.sqrt(np.mean((solution.y[2] - (recovered.values * correction_factor/13500))**2))
	return sol_inf * 0.5 + sol_rec * 0.5

def loss_seir(point, confirmed, recovered, exposed):
	size = len(confirmed)
	beta, gamma, mu, sigma = point
	def model(t, y):
		S = y[0]
		I = y[1]
		R = y[2]
		E = y[3]
		return [mu - beta * S * I - mu * S, beta * S * I - sigma * E - mu * E, sigma * E * I - gamma * I - mu * I, gamma * I - mu * R]
	solution = solve_ivp(model, [0, size], [S_0,E_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)
	sol_inf = np.sqrt(np.mean((solution.y[1] - (confirmed.values.flatten() * correction_factor/13500))**2))
	sol_rec = np.sqrt(np.mean((solution.y[2] - (recovered.values * correction_factor/13500))**2))
	sol_exp = np.sqrt(np.mean((solution.y[3] - (exposed.values * correction_factor/13500))**2))
	return sol_inf/3 + sol_rec/3 + sol_exp/3

my_learner = Learner('US')
my_learner.train()
