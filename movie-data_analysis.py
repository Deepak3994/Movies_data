import pandas as pd

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('/home/deepak/Documents/Deepak/data_analysis/Data for analysis/ml-1m/users.dat', 
                      sep='::', header=None, names=unames, engine='python')
print(users)


rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('/home/deepak/Documents/Deepak/data_analysis/Data for analysis/ml-1m/ratings.dat', 
                        sep='::', header=None, names=rnames, engine='python')
print(ratings)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('/home/deepak/Documents/Deepak/data_analysis/Data for analysis/ml-1m/movies.dat', 
                       sep='::', header=None, names=mnames, engine='python')
print(movies)



data = pd.merge(pd.merge(ratings,users),movies)
print(data)



mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
print(mean_ratings[:20])



rating_of_movie = data.groupby('title').size()
active_titles = rating_of_movie.index[rating_of_movie>=250]
print(active_titles)



mean_ratings = mean_ratings.ix[active_titles]
print(mean_ratings)



top_female_ratings = mean_ratings.sort_index(by='F', ascending=False)
print(top_female_ratings[:10])



mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_index(by='diff')
print(sorted_by_diff[:10])



rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.ix[active_titles]
rating_std_by_title.order(ascending=False)
print(rating_std_by_title[:10])


#Below is the 2 visualization done using matplotlib
'''
get_ipython().magic('matplotlib nbagg')
import matplotlib.pyplot as plt
import numpy as np

my_plot = rating_std_by_title[:10].plot(kind='bar', title='Disagreement among viewers in watching movies')
my_plot.set_xlabel("Movies")
my_plot.set_ylabel("Standard Deviation")



my_plot = mean_ratings[:20].plot(kind='bar', title='Mean ratings of movies by Male & Female')

'''