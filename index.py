from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
app = Flask(__name__, static_folder='prod_price_pred')

model = joblib.load('xgboost_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer_product_rating_xgb.pkl')
@app.route('/')
def index():
    return render_template('Firstpage.html')
@app.route('/predict', methods=['POST'])
def predict_():
    return render_template('dropdown.html')
@app.route('/submit', methods=['POST'])
def submit():
    option_content_mapping1 = {
    '1': 'ELECTRONICS',
    '4': 'SPORTS,BOOKS AND MORE',
    '3': "Men's Wear",
    '5': "Women's Wear",
    '0': 'Baby and Kids',
    '2': 'Home and Furniture'
}
    option_content_mapping2 = {
    '62': 'Smart Wearable Tech',
    '64': 'Health Care Appliances',
    '21': 'Exercise Fitness',
    '33': 'Health and Nutrition',
    '7': 'BOOKS',
    '66': 'Stationary',
    '1': 'Auto Accessories',
    '38': 'Industrial & Scientific tools',
    '51': 'Medical Supplies',
    '28': 'Gaming',
    '53': 'Mobile Accessories',
    '34': 'Health Care Appliances',
    '48': 'Laptops',
    '18': 'Desktop PCs',
    '15': 'Computer Accessories',
    '63': 'Speakers',
    '13': 'Cameras',
    '12': 'Camera Accessories',
    '54': 'Network Components',
    '25': 'Foot Wear',
    '52': "Men's Grooming",
    '72': 'Top Wear',
    '8': 'Bottom Wear',
    '67': 'Suits, Blazers, and Waistcoats',
    '71': 'Ties, Socks, Caps, and More',
    '22': 'Fabrics',
    '76': 'Winter Wear',
    '20': 'Ethnic Wear',
    '39': 'Innerwear and Loungewear',
    '59': 'Raincoats and Windcheaters',
    '74': 'Watches',
    '0': 'Accessories',
    '57': 'Personal Care Appliances',
    '75': 'Western and Maternity Wear',
    '49': 'Lingerie and Sleepwear',
    '56': 'Party Dresses',
    '65': 'Sports Wear',
    '5': 'Beauty and Grooming',
    '41': 'Kids Clothing',
    '9': 'Boys Clothing',
    '30': 'Girls Clothing',
    '2': 'Baby Boy Clothing',
    '4': 'Baby Girl Clothing',
    '42': 'Kids Footwear',
    '10': 'Boys Footwear',
    '31': 'Girls Footwear',
    '44': 'Kids Watches',
    '43': 'Kids Sunglasses',
    '47': 'Kids Winter Wear',
    '73': 'Toys',
    '42': 'School Supplies',
    '3': 'Baby Care',
    '46': 'Kitchen, Cookware, and Serveware',
    '70': 'Tableware and Dinnerware',
    '47': 'Kitchen Storage',
    '14': 'Cleaning Supplies',
    '6': 'Bedroom Furniture',
    '50': 'Living Room Furniture',
    '55': 'Office and Study Furniture',
    '17': 'DIY Furniture',
    '27': 'Furnishing',
    '61': 'Smart Home Automation',
    '36': 'Home Improvement',
    '35': 'Home Decor',
    '37': 'Home Lighting',
    '23': 'Festive Decor and Gifts',
    '58': 'Pet Supplies'
}
    option_content_mapping3 = {
    '242': 'Smart Glasses (VR)',
    'Mobile Cases': 'Mobile Cases',
    '68': 'Cricket',
    '20': 'Badminton',
    '71': 'Cycling',
    '109': 'Football',
    '234': 'Skating',
    '45': 'Camping and Hiking',
    '264': 'Swimming',
    '49': 'Cardio Equipment',
    '136': 'Home Gyms',
    '259': 'Support',
    '90': 'Dumbbells',
    '0': 'Ab Exercisers',
    '293': 'Shakers and Sippers',
    '297': 'Yoga Mat',
    '126': 'Gym Gloves',
    '187': 'Nuts and Dry Fruits',
    '299': 'Snacks and Beverages',
    '58': 'Chocolates',
    '121': 'Gifting Combos',
    '262': 'Sweets Store',
    '204': 'Protein Supplements',
    '285': 'Vitamin Supplements',
    '132': 'Health Drinks',
    '5': 'Ayurvedic Supplements',
    '': 'Entrance Exams',
    '1': 'Academics',
    '167': 'Literature and Fiction',
    '185': 'Non Fiction',
    '298': 'Young Readers',
    '221': 'Self-Help',
    '197': 'Pens',
    '81': 'Diaries',
    '48': 'Card Holders',
    '76': 'Desk Organizers',
    '43': 'Calculators',
    '146': 'Key Chains',
    '135': 'Helmets and Riding Gears',
    '46': 'Car Audio/Video',
    '47': 'Car Mobile Accessories',
    '283': 'Vehicle Lubricants',
    '140': 'Industrial Measurement Devices',
    '': 'Industrial Testing Devices',
    '159': 'Lab and Scientific Products',
    '189': 'Packaging and Shipping Products',
    '212': 'Safety Products',
    '201': 'Pregnancy and Fertility Kits',
    '138': 'Hot Water Bag',
    '117': 'Gaming Consoles',
    '116': 'Gaming Accessories',
    '241': 'Smart Glasses (VR)',
    '177': 'Mobile Cases',
    '131': 'Headphones & Headsets',
    '199': 'Power Banks',
    '220': 'Screenguards',
    '175': 'Memory Cards',
    '243': 'Smart Headphones',
    '176': 'Mobile Cables',
    '178': 'Mobile Chargers',
    '179': 'Mobile Holders',
    '245': 'Smart Watches',
    '242': 'Smart Glasses',
    '39': 'Bp Monitors',
    '274': 'Weighing Scale',
    '118': 'Gaming Laptops',
    '77': 'Desktop PCs',
    '99': 'External Hard Disks',
    '196': 'Pendrives',
    '161': 'Laptop Skins and Decals',
    '160': 'Laptop Bags',
    '181': 'Mouse',
    '203': 'Printers and Ink Cartridges',
    '180': 'Monitors',
    '3': 'Apple iPads',
    '251': 'Soundbars',
    '33': 'Bluetooth Speakers',
    '73': 'DTH Set Top Box',
    '72': 'DSLR and Mirrorless',
    '255': 'Sports and Action',
    '155': 'Lens',
    '281': 'Tripods',
    '209': 'Routers',
    '253': 'Sports Shoes',
    '54': 'Casual Shoes',
    '111': 'Formal Shoes',
    '215': 'Sandals and Floaters',
    '107': 'Flip-Flops',
    '168': 'Loafers',
    '35': 'Boots',
    '210': 'Running Shoes',
    '246': 'Sneakers',
    '74': 'Deodorants',
    '198': 'Perfumes',
    '27': 'Beard Care and Grooming',
    '227': 'Shaving and Aftershave',
    '222': 'Sexual Wellness',
    '265': 'T-Shirts',
    '110': 'Formal Shirts',
    '53': 'Casual Shirts',
    '143': 'Jeans',
    '55': 'Casual Trousers',
    '112': 'Formal Trousers',
    '278': 'Track pants',
    '231': 'Shorts',
    '50': 'Cargos',
    '271': 'Three Fourths',
    '257': 'Sweatshirts',
    '142': 'Jackets',
    '260': 'Sweater',
    '279': 'Tracksuits',
    '156': 'Kurta',
    '96': 'Ethnic Sets',
    '228': 'Sherwan'
}


    category1 = request.form.get('category1')
    category2 = request.form.get('category2')
    category3 = request.form.get('category3')
    productTitle = request.form.get('productTitle')
    mrp = request.form.get('mrp')
    
    input_data = [{
        'title': productTitle,
        'category_1': -int(category1),
        'category_2': -int(category2),
        'category_3': -int(category3),
        'mrp': int(mrp)
    }]
    
    text_data = input_data[0]['title']
    X_text_input_transformed = vectorizer.transform([text_data])
    X_input = np.hstack((X_text_input_transformed.toarray(), 
                         np.array([[input_data[0]['category_1'], input_data[0]['category_2'], input_data[0]['category_3'], input_data[0]['mrp']]] * X_text_input_transformed.shape[0])))
    
    predicted_price = model.predict(X_input)[0]
    
    chosen_category_redirect = " ".join([option_content_mapping1[str(int(category1))], option_content_mapping2[str(int(category2))], option_content_mapping3[str(int(category3))]])

    return render_template('result.html', predicted_price=predicted_price, product_title=productTitle, chosen_category=chosen_category_redirect)
if __name__ == '__main__':
    app.run(port=5000)
