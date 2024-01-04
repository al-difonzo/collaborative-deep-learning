AMZ_EMBEDDED_COLS = ['fullReview','itemMetadata']

AMZ_CHOICES = [
    'Amazon Fashion',
    'All Beauty',
    'Appliances',
    'Arts, Crafts and Sewing',
    'Automotive',
    'Books',
    'CDs and Vinyl',
    'Cell Phones and Accessories',
    'Clothing, Shoes and Jewelry',
    'Digital Music',
    'Electronics',
    'Gift Cards',
    'Grocery and Gourmet Food',
    'Home and Kitchen',
    'Industrial and Scientific',
    'Kindle Store',
    'Luxury Beauty',
    'Magazine Subscriptions',
    'Movies and TV',
    'Musical Instruments',
    'Office Products',
    'Patio, Lawn and Garden',
    'Pet Supplies',
    'Prime Pantry',
    'Software',
    'Sports and Outdoors',
    'Tools and Home Improvement',
    'Toys and Games',
    'Video Games',
]

AMZ_CHOICES_ = [name.replace(',','').replace(' and ','_').replace(' ','_') for name in AMZ_CHOICES]

# 'source_zip': zip name as reported at https://grouplens.org/datasets/movielens/
# 'UIMat_file': path of file containing User-Item Matrix after extracting zip contents
MOVIELENS_CHOICES = {
    'ml-lat-small': {
        'source_zip': 'ml-latest-small.zip',
        'UIMat_file': 'ml-latest-small/ratings.csv',
        'cols_to_embed': ['combined_info'],
    },
}

ML_BASE_URL = 'http://files.grouplens.org/datasets/movielens'

TMDB_BASE_URL = 'https://api.themoviedb.org/3/movie'