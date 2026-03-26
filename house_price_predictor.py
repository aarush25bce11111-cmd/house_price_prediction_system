import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

def main():
    print("--- House Price Predictor Project ---")
    print("Initializing synthetic dataset...\n")
    data = {
        'SquareFootage': [1500, 2000, 2500, 1200, 1800, 3000, 1400, 2200, 1100, 2700,
                          1600, 1900, 2400, 1300, 2800, 3200, 1450, 2100, 1150, 2600],
        'Bedrooms': [3, 4, 4, 2, 3, 5, 2, 4, 2, 4,
                     3, 3, 4, 2, 5, 5, 3, 4, 2, 4],
        'Age': [10, 5, 2, 20, 15, 1, 25, 8, 30, 3,
                12, 6, 4, 18, 2, 1, 22, 7, 28, 5],
        'Location': ['Suburb', 'City', 'City', 'Rural', 'Suburb', 'City', 'Rural', 'Suburb', 'Rural', 'City',
                     'Suburb', 'Suburb', 'City', 'Rural', 'City', 'City', 'Suburb', 'Suburb', 'Rural', 'City'],
        'Price': [250000, 350000, 420000, 180000, 290000, 500000, 195000, 360000, 160000, 450000,
                  265000, 310000, 400000, 185000, 480000, 520000, 240000, 345000, 170000, 440000]
    }
    df = pd.DataFrame(data)
    df.loc[2, 'SquareFootage'] = np.nan
    df.loc[5, 'SquareFootage'] = np.nan
    print("Raw Data Sample (with missing values):")
    print(df.head(6))
    imputer = SimpleImputer(strategy='mean')
    df['SquareFootage'] = imputer.fit_transform(df[['SquareFootage']])
    print("\nData after cleaning (missing values filled with mean):")
    print(df.head(6))
    df = pd.get_dummies(df, columns=['Location'], drop_first=True) 
    print("\nData after encoding (Location converted to numbers):")
    print(df.head())
    print("-" * 30)
    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nModel trained successfully!")
    print(f"Intercept (Base Price): ${model.intercept_:.2f}")
    print("Coefficients (Weight of each feature):")
    for feature, coef in zip(X.columns, model.coef_):
        print(f" - {feature}: {coef:.2f}")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n--- Model Evaluation ---")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.4f} (Closer to 1.0 is better)") 
    print("\n--- Live Prediction (Interactive) ---")
    try:
        print("Please enter details for the house you want to price:")
        user_sqft = float(input("   Square Footage (e.g., 2000): "))
        user_beds = float(input("   Number of Bedrooms (e.g., 3): "))
        user_age = float(input("   Age of House in years (e.g., 5): "))
        user_loc = input("   Location (City, Suburb, or Rural): ").strip().lower()
        loc_rural = 0
        loc_suburb = 0
        if user_loc == 'rural':
            loc_rural = 1
        elif user_loc == 'suburb':
            loc_suburb = 1
        elif user_loc != 'city':
            print("   (Unknown location entered, defaulting to 'City' pricing)")
        new_house = np.array([[user_sqft, user_beds, user_age, loc_rural, loc_suburb]]) 
        predicted_price = model.predict(new_house)
        print(f"\nEstimated Price: ${predicted_price[0]:,.2f}")
    except ValueError:
        print("\nError: Please enter valid numbers for square footage, bedrooms, and age.")
    try:
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2) # Diagonal line
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted House Prices')
        print("\n(Graph generated - check window if available)")
        plt.show()
    except Exception as e:
        print("\n(Could not generate graph in this environment)")

if __name__ == "__main__":
    main()
    