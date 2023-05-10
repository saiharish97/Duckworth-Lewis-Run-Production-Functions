"""
This script performs data cleaning and analysis on cricket match data
to compute the resources remaining using the Duckworth-Lewis method.
Author: Pathuri Sai Harish
Date: May 10, 2023
"""
import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import sys


def data_clean(data):

    data = data[data.Innings == 1]

    # Consider only those games which were completed and not interrupted by rain
    # This means the games that were played 50 overs with atleast 1 wicket remaining
    # and < 50 overs with 0 wickets remaining.

    needed_matches = pd.DataFrame(
        {
            'played_50_overs' : data.groupby(data['Match'])['Over'].count() == 50,
            'all_out': data.groupby(data['Match'])['Wickets.in.Hand'].min() == 0
        }
        ).reset_index()
    #select rows which have atleast one of the columns True
    needed_matches['needed_matches'] = needed_matches['played_50_overs'] | needed_matches['all_out']
    needed_matches_list = list(needed_matches[needed_matches['needed_matches'] == True].Match)
    include_index = []
    for i in needed_matches_list:
        include_index += list(data[data.Match == i].index)
    data = data.loc[include_index, :]
    data['Overs.Remaining'] = 50 - data['Over']
    # Create dataframe consisting columns - 'Match', 'Runs.Remaining', 'Wickets.in.Hand','Overs.Remaining', 'Innings.Total.Runs' 
    data = pd.DataFrame(data, columns=['Match', 'Runs.Remaining', 'Wickets.in.Hand','Overs.Remaining', 'Innings.Total.Runs'])
    return data, needed_matches_list


def mean_max_runs_at_given_wicket(data, w):
    data = data[data['Wickets.in.Hand'] == w]
    max_runs = data.groupby(['Match'])['Runs.Remaining'].max()
    return np.mean(max_runs)

class DLModel():

    def __init__(self,parameters, matches_list, train_runs,train_wickets,train_overs,data_util,constraints):
        self.parameters=parameters
        self.matches_list=matches_list
        self.train_runs=train_runs
        self.train_wickets=train_wickets
        self.train_overs=train_overs
        self.data_util=data_util
        self.constraints=constraints


    def Z_function(self,Z_0, L, u):
        return Z_0 * (1 - np.exp(-L * u / Z_0))

    # Function to compute the error function per wicket
    def error_function(self,parameters):
        Z0 = parameters[:10]
        L = parameters[10]
        mse = 0
        for i in range(len(self.train_runs)):
            z = self.Z_function(Z0[self.train_wickets[i]-1], L, self.train_overs[i])
            mse += (z - self.train_runs[i])**2
        mse /= len(self.train_runs)
        # The 50 overs remaining data points are not present in the dataframe.
        # So, manually looping over the Innings Total Runs for loss correction of each match will suffice
        mse_correction = 0
        for i in self.matches_list:
            runs = self.data_util.get_match_total_runs(i)
            z = self.Z_function(Z0[9], L, 50)
            mse_correction += (z - runs)**2
        mse_correction /= len(self.matches_list)
        mse += mse_correction
        return mse
    
    def optimize_error(self):
        # methods = ['COBYLA', 'SLSQP', 'trust-constr']
        # MSE ->  :  COBYLA   ->  5077.119336230687            
        # MSE ->  : -> SLSQP  4989.935425213922            <---------------------------
        # MSE ->  : trust-constr  ->  4994.192160361856
       
        methods=["SLSQP"]
        
        min_mse= sys.maxsize
        best_method=None
        for m_type in methods:
            print("Trying to fit through " + m_type)
            opter = opt.minimize(self.error_function, self.parameters, method=m_type, constraints=self.constraints)
            print("MSE -> " +m_type +" : -> " + str(opter.fun))
            if min_mse>opter.fun:
                best_method=m_type
                min_mse=opter.fun
                L_out = opter.x[-1]
                Z0_out = opter.x[:-1]
        print("Best Minimization Algo Method: "+best_method)
        print("Z0s: ")
        print(Z0_out)
        print("L: ")
        print(L_out)
        return Z0_out, L_out


class DataUtil():
     
    def __init__(self, data):
         self.data=data

    def get_match_total_runs(self, id):
        return list(self.data[(self.data['Match'] == id)]['Innings.Total.Runs'])[0]
    
    def plot_the_graphs(self, x, y, title, x_label, y_label, file_name):
        plt.figure()
        for i in range(10):
            plt.plot(x, y[i])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim((0, 50))
        plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        plt.grid()
        plt.savefig(file_name)
        # plt.show()

if __name__=="__main__":
    data = pd.read_csv(f"../data/04_cricket_1999to2011.csv")
    cleaned_data, match_list=data_clean(data)
    print(cleaned_data)

    train_runs = cleaned_data['Runs.Remaining'].values
    train_overs = cleaned_data['Overs.Remaining'].values
    train_wickets = cleaned_data['Wickets.in.Hand'].values

    # Initialize the values of Z0 and L
    parameters = []
    for i in range(1,11):
        parameters.append(mean_max_runs_at_given_wicket(cleaned_data, i))
    parameters.append(7)

    z0_constraints = []
    for i in range(9):
        z0_constraints.append({'type': 'ineq', 'fun': lambda x: x[i+1] - x[i]})
    z0_constraints.append({'type': 'ineq', 'fun': lambda x: x[0]})
    l_constraint = {'type': 'ineq', 'fun': lambda x: x[-1]}
    constraints = z0_constraints + [l_constraint]

    data_util=DataUtil(cleaned_data)
    model=DLModel(
        parameters=parameters,
        matches_list=match_list,
        train_runs=train_runs,
        train_wickets=train_wickets,
        train_overs=train_overs,
        data_util=data_util,
        constraints=constraints
    )
    Z0,L=model.optimize_error()
    u = np.linspace(0, 50, num=300)
    Z_final = []
    for i in range(10):
        Z_final.append(model.Z_function(Z0[i], L, u))

    data_util.plot_the_graphs(u,Z_final,"Average Runs Obtainable through DL Method", "Overs Remaining", "Average Runs","average_runs.png")

    P_final = []
    for i in range(10):
        P_final.append(model.Z_function(Z0[i], L, u)*100/model.Z_function(Z0[9], L, 50))
    
    data_util.plot_the_graphs(u,P_final,"Resources Remaining through DL Method", "Overs Remaining", "Resources Remaining","resources_remaining.png")