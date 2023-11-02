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