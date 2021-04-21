import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import KNNBasic
from surprise import SlopeOne
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
import gc

def plot_accuracies(x_values, model_names, num, labels, title, x_axis):
    # set width of bar
    width = 0.75 / num
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set position of bar on X axis
    br = []
    br.append(np.arange(len(x_values[0])))
    for i in range(len(x_values) - 1):
        br.append([x + width for x in br[i]])
    
    rects = []
    # Make the plot
    for i in range(len(x_values)):
        rects.append(plt.bar(br[i],x_values[i],width=width,edgecolor='grey',label=labels[i]))
    
    # Adding Xticks
    plt.xlabel('Model', fontweight ='bold', fontsize = 15)
    plt.ylabel(x_axis, fontweight ='bold', fontsize = 15)
    plt.xticks([r + 0.25 for r in range(len(x_values[0]))],model_names)

    font = 10
    for rect_ in rects:
        for rect in rect_:
            height = rect.get_height()
            ax.annotate('{:.5f}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=font)
    
    plt.title(title)
    plt.legend(loc='best')

    #fig.savefig('comparison%02i_unbanded_accuracy_ada_rf.png' % 1)
    plt.show()


def plot_accuracies_c_d(x_values, model_names, num, labels, title, x_axis):
    # set width of bar
    width = 0.75 / num
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set position of bar on X axis
    br = []
    br.append(np.arange(len(x_values[0])))
    for i in range(len(x_values) - 1):
        br.append([x + width for x in br[i]])
    
    rects = []
    # Make the plot
    for i in range(len(x_values)):
        rects.append(plt.bar(br[i],x_values[i],width=width,edgecolor='grey',label=labels[i]))
    
    # Adding Xticks
    plt.xlabel('Model', fontweight ='bold', fontsize = 15)
    plt.ylabel(x_axis, fontweight ='bold', fontsize = 15)
    plt.xticks([r + 0.185 for r in range(len(x_values[0]))],model_names)

    font = 8
    for rect_ in rects:
        for rect in rect_:
            height = rect.get_height()
            ax.annotate('{:.3f}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=font)
    
    plt.title(title)
    plt.legend(loc='best')

    #fig.savefig('comparison%02i_unbanded_accuracy_ada_rf.png' % 1)
    plt.show()

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)
ratings = Dataset.load_from_file('ratings_small.csv',reader=reader)

pmf = SVD()
sim_options_item = {'user_based': False}
sim_options_user = {'user_based': True}
ubcf = KNNBasic(sim_options=sim_options_user) # Use KNN
ibcf = KNNBasic(sim_options=sim_options_item) # Use KNN

# Compare RMSE and MAE (Questions c and d)
if False:
    models = [pmf,ubcf,ibcf]
    labels = ['PMF','UBCF','IBCF']
    results = [[],[]]

    for label,rs in zip(labels,models):    
        print('========================================================================')
        print('{}'.format(label))
        print('========================================================================')
        result = cross_validate(rs, ratings, measures=['RMSE','MAE'], cv=5, verbose=True)
        results[0].append(result['test_rmse'].mean())
        results[1].append(result['test_mae'].mean())
        print()

    plot_accuracies_c_d(results, labels, 2,['RMSE','MAE'], 'Comparison of Error Measurement', 'Error')

# Compare Cosine, MSD, and Pearson similarities (Question e)
if False:
    sim_options_item = {'user_based': False}
    sim_options_user = {'user_based': True}
    ubcf = KNNBasic(sim_options=sim_options_user) # Use KNN
    ibcf = KNNBasic(sim_options=sim_options_item) # Use KNN

    sims = ['Cosine', 'MSD', 'Pearson']
    models = [ubcf,ibcf]
    labels = ['UBCF', 'IBCF']
    results_rmse = [[],[],[]]
    results_mae = [[],[],[]]

    for idx, sim in enumerate(sims):
        sim_options_item = {'name': sim, 'user_based': True}
        sim_options_user = {'name': sim, 'user_based': False}
        for label,rs in zip(labels,models):    
            print('========================================================================')
            print('{}'.format(label))
            print('========================================================================')
            result = cross_validate(rs, ratings, measures=['RMSE','MAE'], cv=5, verbose=True)
            results_rmse[idx].append(result['test_rmse'].mean())
            results_mae[idx].append(result['test_mae'].mean())
            print()

    plot_accuracies(results_rmse, labels, 3, sims, 'Comparison of Similarities', 'RMSE')
    plot_accuracies(results_mae, labels, 3, sims, 'Comparison of Similarities', 'MAE')

# Compare K (Questions f and g)
if True:    
    if False:
        K_ib = [i for i in range(0,100,10)]
        K_us = [i for i in range(0,100,10)]
        K_all = [K_ib,K_us]
        sim_options_item = {'user_based': False}
        sim_options_user = {'user_based': True}
        labels = ['IBCF', 'UBCF']
        model = KNNBasic
        options = [sim_options_item,sim_options_user]

        for label,option,K in zip(labels,options,K_all):     
            results_rmse = []
            results_mae = []
            print('========================================================================')
            print('{}'.format(label))
            print('========================================================================')
            for idx, k in enumerate(K):        
                rs = model(k=k,k_min=k,sim_options=option)
                result = cross_validate(rs, ratings, measures=['RMSE'], cv=5, verbose=True)
                results_rmse.append(result['test_rmse'].mean())
                print()
                
            fig, ax = plt.subplots(figsize=(12, 8))
            
            K_ = [str(k) for k in K]
            # creating the bar plot
            rect_ = plt.plot(K_, results_rmse)
            
            plt.xlabel("K Value")
            plt.ylabel("RMSE")
            plt.title("Comparison of Differnt K Values with {}".format(label))
            plt.show()

    if True:
        K_ib = [i for i in range(70,90,2)]
        K_us = [i for i in range(5,30)]
        K_all = [K_ib,K_us]
        sim_options_item = {'user_based': False}
        sim_options_user = {'user_based': True}
        labels = ['IBCF', 'UBCF']
        model = KNNBasic
        options = [sim_options_item,sim_options_user]

        for label,option,K in zip(labels,options,K_all):     
            results_rmse = []
            results_mae = []
            print('========================================================================')
            print('{}'.format(label))
            print('========================================================================')
            for idx, k in enumerate(K):        
                rs = model(k=k,k_min=k,sim_options=option)
                result = cross_validate(rs, ratings, measures=['RMSE'], cv=5, verbose=True)
                results_rmse.append(result['test_rmse'].mean())
                print()
                
            fig, ax = plt.subplots(figsize=(12, 8))
            
            K_ = [str(k) for k in K]
            # creating the bar plot
            rect_ = plt.plot(K_, results_rmse)
            
            plt.xlabel("K Value")
            plt.ylabel("RMSE")
            plt.title("Comparison of Differnt K Values with {}".format(label))
            plt.show()