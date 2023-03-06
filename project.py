#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:25:51 2021

@author: brandon
"""

import json
import pandas as pd
import numpy as np
import pymc3 as pm
from pymc3 import BetaBinomial, Exponential, Model, sample
import arviz as az
from scipy.stats import betabinom, beta, binom, truncnorm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
from os.path import exists
import math

base_dir = '/home/brandon/Dropbox/coursework/bayes/project/'

MAX_STARS=5
MIN_STARS=1
NUM_REVIEWS=20

def rating_to_prob(stars):
    #return (stars-MIN_STARS) / MAX_STARS + 0.01
    return stars-MIN_STARS

def save_json(key, obj):
    data_file = open(base_dir + 'cache/' + key + '.json', 'w')
    json.dump(obj, data_file)

def load_json(key):
    data_file_name = base_dir + 'cache/' + key + '.json'
    if not exists(data_file_name):
        return None
    data_file = open(data_file_name)
    return json.load(data_file)

def savefig(title):
    plt.savefig(base_dir + 'plots/' + title + '.png',
            bbox_inches='tight', pad_inches=0.1)
    plt.close()

def load_reviews():
    movie_reviews = load_json('movie_reviews')
    all_ratings = load_json('all_ratings')
    if movie_reviews is not None and all_ratings is not None:
        return movie_reviews, all_ratings
    reviews_df = pd.read_csv(base_dir + '/ml-latest-small/ratings.csv')
    reviews_df['rating'].hist(density=True)
    reviews_df['rating'] = (reviews_df['rating'] - 0.5).astype(int)
    
    all_ratings = []
    movie_dict = {}
    for index, review in reviews_df.iterrows():
        movie_id = int(review['movieId'])
        rating = int(review['rating'])
        all_ratings.append(rating)
        if movie_id in movie_dict:
            movie_dict[movie_id].append(rating)
        else:
            movie_dict[movie_id] = [rating]
    
    movie_reviews = []
    for movie, reviews in movie_dict.items():
        if len(reviews) > NUM_REVIEWS:
            movie_reviews.append((movie, reviews))
    
    print('#movies with >= 20 reviews', len(movie_reviews))
    save_json('movie_reviews', movie_reviews)
    save_json('all_ratings', all_ratings)
    return movie_reviews, all_ratings

movie_reviews, all_ratings = load_reviews()

stars_np = np.array(all_ratings)
stars_np = np.random.choice(stars_np, (1000))
#print(stars_np)

X = np.arange(MAX_STARS)
print('X', X)
Y = np.bincount(stars_np) / stars_np.size
print('Y', Y)

plt.figure()
plt.hist(all_ratings, bins=len(Y))
savefig('All review histogram')

def plot_binomial(p, ratings, title):
    print(title, ratings)
    n = MAX_STARS #obs.max() + 1
    X = np.arange(n)
    ratings = np.array(ratings)
    Y = np.bincount(ratings.astype(int), minlength=n) / ratings.size
    print('X', X)
    print('Y', Y)
    
    plt.figure()
    plt.hist(ratings, bins=len(Y))
    plt.show()
    
    # From https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
    fig, ax = plt.subplots(1, 1)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(X + MIN_STARS, Y, 'bo', ms=8, label='review count')
    ax.vlines(X + MIN_STARS, 0, Y, colors='b', lw=5, alpha=0.5)
    Y_ = binom.pmf(X, n, p)
    print(Y_)
    mse = ((Y - Y_)**2).sum() / MAX_STARS
    print('Mean squared error: %0.4f' % mse)

    plt.xlabel('Rating')
    plt.ylabel('Proportion of reviews')
    plt.title(title + ' mse=%0.4f' % mse)
    ax.plot(X + MIN_STARS, Y_, label='Binomial model')
    plt.legend()
    savefig(title)
    
    return mse
   

def run_binomial(obs, title):
    cached_results = load_json(title + '_binomial')
    if cached_results is not None:
        return cached_results
    print(title, obs)
    n = MAX_STARS #obs.max() + 1
    
    with Model() as model:
        p = pm.Uniform('p', 0, 1)
    
        x = pm.Binomial('rating', n=n, p=p, observed=obs)

        step = pm.NUTS()
    
        trace = pm.sample(10000, tune=1000,
                       step=step,
                       #target_accept=0.9,
                       cores=8,
                       init='auto',
                       return_inferencedata=True)
        print(trace)
        az.plot_trace(trace)
        print(az.summary(trace, hdi_prob=0.95))
        p_mean = trace.posterior['p'].mean().values.tolist()
        print('p mean', p_mean)
        
    results = p_mean, plot_binomial(p_mean, obs, title)
    save_json(title + '_binomial', results)
    return results

def get_obs_from_file(file_name):
    print(file_name)
    obs = []
    with open(base_dir + file_name, 'rt') as file:
        for line in file.readlines():
            obs.append(int(line.strip()) - 1)
    print(obs)
    return np.array(obs)

#p = run_binomial(stars_np, 'All reviews')
movie_results = load_json('movie_results')
if movie_results is None:
    movie_results = {}
for movie, reviews in movie_reviews:
    movie = str(movie)
    if len(reviews) < NUM_REVIEWS:
        continue
    if movie in movie_results:
        print('Loading', movie, movie_results[movie])
        continue
    np_reviews = np.array(reviews)
    #params, mse = run_truncated_normal(np_reviews, 'id-' + movie + '-normal')
    p, mse = run_binomial(np_reviews, 'id-' + movie)
    movie_results[movie] = (p, mse, np_reviews.mean(), np_reviews.std())
    print('Saving', movie, p, mse, np_reviews.mean(), np_reviews.std())
    save_json('movie_results', movie_results)

all_mse = []
for movie, (p, mse, mu, sigma) in movie_results.items():
    all_mse.append(mse)
all_mse = np.array(all_mse)
print('Average mse', all_mse.mean())

plt.figure()
plt.title('MSE across all movies')
plt.hist(all_mse, bins=50)
savefig('MSE across all movies')

from cmdstanpy import cmdstan_path, CmdStanModel

def run_censored_binomial(ratings, right_censored, title):
    n = MAX_STARS
    data = {
        'n': n,
        'M': len(ratings),
        'ratings': ratings,
        'right_censored': right_censored
        }
    save_json('censored_binomial', data)
    
    stan = base_dir + 'censored_binomial.stan'
    model = CmdStanModel(stan_file=stan)
    model.name
    model.stan_file
    model.exe_file
    model.code()
    
    data = base_dir + 'cache/censored_binomial.json'
    fit = model.sample(data=data,
                       iter_warmup=1000,
                       adapt_delta=0.90,
                       iter_sampling=30000,
                       output_dir=base_dir + 'stan_temp_out')
    
    summary = fit.summary()
    print('###############summary')
    print(summary)
    #print('###########diagnose')
    #print(fit.diagnose())
    p = summary['Mean']['p']
    
    return p

def simulate_thermostat_ratings(title, reviews, limit=NUM_REVIEWS):
    cached_results = load_json(title + '_thermostat_ratings')
    if cached_results is not None:
        return cached_results
    relative_results = []
    p = 0.5
    curr_ratings = []
    right_censored = []
    print('limit', limit)
    for j in range(limit):
        expected_rating = p * MAX_STARS
        if expected_rating < reviews[j]:
            curr_ratings.append(math.ceil(expected_rating))
            right_censored.append(1)
        else:
            curr_ratings.append(math.floor(expected_rating))
            right_censored.append(0)
        
        print(movie, j, expected_rating, reviews[j], curr_ratings[-1], right_censored[-1])
        p = run_censored_binomial(curr_ratings, right_censored, f'{title}-relative-{j}')
        estimated_mean_rating = p * MAX_STARS
        rolling_window = curr_ratings
        if len(rolling_window) > 10:
            rolling_window = rolling_window[-10:]
        actual_mean_rating = np.array(rolling_window).mean()
        relative_results.append((estimated_mean_rating, actual_mean_rating))
        
    save_json(title + '_thermostat_ratings', relative_results)
    return relative_results

def plot_relative_ratings(title, results):
    estimated, actual = zip(*results)
    x = np.arange(len(estimated))
    plt.figure()
    ax = plt.gca()
    ax.set_ylim([0, 5])
    plt.title(title)
    plt.xlabel('Review Count')
    plt.ylabel('Rating')
    points = np.full((len(estimated)), 6)
    
    upper_x = []
    upper_y = []
    lower_x = []
    lower_y = []
    for i in range(1, len(x)):
        if estimated[i] > estimated[i-1]:
            upper_x.append(x[i-1])
            upper_y.append(estimated[i-1])
        else:
            lower_x.append(x[i-1])
            lower_y.append(estimated[i-1])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x, np.array(actual) + 1, label='Actual')
    plt.plot(x, np.array(estimated) + 1, label='Binomial model')
    plt.errorbar(upper_x, np.array(upper_y) + 1, xerr=0.4, yerr=0.5,
                 lolims=True, linestyle='', alpha=0.5)
    plt.errorbar(lower_x, np.array(lower_y) + 1, xerr=0.4, yerr=0.5,
                 uplims=True, linestyle='', alpha=0.5)
    plt.legend()
    savefig(title)

relative_results = load_json('relative_results')
if relative_results is None:
    relative_results = {}

def compute_relative_results():
    for movie, reviews in movie_reviews:
        movie_key = str(movie)
        if movie_key not in relative_results:
            relative_results[movie_key] = simulate_thermostat_ratings(movie_key, reviews)
            save_json('relative_results', relative_results)
            
            print('***************************')
            print('RELATIVE_RESULTS', relative_results[movie_key])
            print('***************************')

def compute_average_results():
    n = len(relative_results)
    error = np.zeros((n, NUM_REVIEWS))
    i = 0
    for movie_key, results in relative_results.items():
        np_results = np.array(results)
        actual_average = np_results[:, 0]
        estimated_average = np_results[:, 1]
        error[i] = (actual_average - estimated_average) ** 2
        i += 1
    mse = np.mean(error, axis=0)
    std = np.std(error, axis=0)
    print('mse', mse)
    print('std', std)
    x = np.arange(len(mse))
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    title = f'MSE across {n} movies given {NUM_REVIEWS} reviews'
    plt.title(title)
    plt.xlabel('Count of reviews')
    plt.ylabel('Average MSE')
    plt.fill_between(x, mse - std, mse + std, alpha=0.2)
    plt.plot(x, mse, label='Number of reviews')
    savefig(title)

def canonical_results():
    titles_by_error = []
    for movie_key, results in relative_results.items():
        np_results = np.array(results)
        actual_average = np_results[:, 0]
        estimated_average = np_results[:, 1]
        error = ((actual_average - estimated_average) ** 2).sum() / NUM_REVIEWS
        titles_by_error.append((error, movie_key))

    titles_by_error.sort()
    N = len(titles_by_error)
    for i in [0, math.ceil(N/4), int(N/2), int(3*N/4), N-1]:
        mse, movie_key = titles_by_error[i]
        plot_relative_ratings('Movie %s: mse=%0.2f (%d%%)' % 
                              (movie_key, mse, int(100 * i / (N-1))),
                              relative_results[movie_key])

compute_relative_results()
canonical_results()

isye_quality_obs = get_obs_from_file('isye_6420_ratings.txt')
isye_difficulty_obs = get_obs_from_file('isye_6420_difficulty.txt')
run_binomial(isye_quality_obs, 'ISYE 6420 OMSCS Quality Ratings Summary')
run_binomial(isye_difficulty_obs, 'ISYE 6420 OMSCS Difficult Ratings Summary')
isye_quality_results = simulate_thermostat_ratings('ISYE Quality',
                                                   isye_quality_obs,
                                                   len(isye_quality_obs))
plot_relative_ratings('ISYE 6420 OMSCS Quality Ratings Relative',
                      isye_quality_results)
isye_difficulty_results = simulate_thermostat_ratings('ISYE Difficulty',
                                                      isye_difficulty_obs,
                                                      len(isye_difficulty_obs))
plot_relative_ratings('ISYE 6420 OMSCS Difficult Ratings Relative',
                      isye_difficulty_results)

compute_average_results()

