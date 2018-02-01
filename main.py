import psycopg2 as pg
import pandas as pd
import numpy as np

# Simple movie recommender system using basic matrix factorization.
# Data from MovieLens Project

def load_data():
    """
    Connect to postgres database containing movielens data. 
    Then load the data into pandas df.
    :return: movies and ratings dataframes
    """
    try:
        conn = pg.connect("dbname='movielens' user='master'")
    except:
        print "I am unable to connect to the db."

    # load sql tables into pandas df
    movies_df = pd.read_sql("SELECT * FROM movies", conn)
    movies_df = movies_df.iloc[1:]
    ratings_df = pd.read_sql("SELECT * FROM ratings", conn)
    ratings_df = ratings_df.astype('int64')
    movies_df['movieid'] = movies_df['movieid'].apply(pd.to_numeric)

    return ratings_df, movies_df

def matrix_SVD(ratings_df):
    """
    dfs look good, but format of ratings matrix should be one row per user and one column per movie.
    We also normalize the ratings matrix by subtracting each users mean rating.
    Using Scipy SVD function since I can choose # of latent factors. 
    We convert the values in sigma into a diagonal matrix to get predictions
    Re-add user means to get predicted 5 star ratings
    :param preds_df:
    :return:  
    """
    from scipy.sparse.linalg import svds

    R_df = ratings_df.pivot(index = 'userid', columns ='movieid', values = 'rating').fillna(0)
    R_mat = R_df.as_matrix()
    user_ratings_mean = np.mean(R_df, axis = 1)
    R_norm = R_mat - user_ratings_mean.values.reshape(-1, 1)

    U, sigma, Vt = svds(R_norm, k = 50)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.values.reshape(-1, 1)
    p_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)

    return p_df

def user_top_movies(userid, movies_df, original_ratings_df):
    """
    Function merges user data with movie information, then sort by the top ratings. 
    :param userid: 
    :param movies_df: 
    :param original_ratings_df: 
    :return: df with user rating and movie data, sorted by top ratings given by the user. 
    """
    # merge user's data with movie info, sort by top ratings
    user_data = original_ratings_df[original_ratings_df.userid == (userid)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieid', right_on = 'movieid'))
    sort_user_full = user_full.sort_values(['rating'], ascending=False)

    return sort_user_full


def recommend_movies(sort_user_full, predictions_df, userid, movies_df, num_recommendations):
    """
    Recommends unseen movies to a specified user. Does not excplictly use movie conent features,
    however SVD picked up on latent user preferences. 
    :param predictions_df: 
    :param userid: 
    :param movies_df: 
    :param num_recommendations: 
    :return: df with top rated unseen movies for the user
    """
    user_row_number = userid - 1
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    recommendations = (movies_df[~movies_df['movieid'].isin(sort_user_full['movieid'])].
        merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
              left_on='movieid',
              right_on='movieid').
            rename(columns={user_row_number: 'Predictions'}).
            sort_values('Predictions', ascending=False).
            iloc[:num_recommendations, :-1]
        )
    return recommendations


def print_recommendations(user_id, num_recommendations, user_rated, recommended_movies):
    """
    Prints headers for both user_top df and recommendations df.
    """
    print 'User {0} has already rated {1} movies.'.format(user_id, user_rated.shape[0])
    print user_rated.head()
    print '\nRecommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations)
    print recommended_movies

if __name__ == '__main__':
    user_id = 100
    num_recommendations = 10

    ratings_df, movies_df = load_data()
    preds_df = matrix_SVD(ratings_df)
    user_top = user_top_movies(user_id, movies_df, ratings_df)
    predictions = recommend_movies(user_top, preds_df, user_id, movies_df, num_recommendations)
    print_recommendations(user_id, num_recommendations, user_top, predictions)





