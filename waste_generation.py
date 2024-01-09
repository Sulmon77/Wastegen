import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the dataset
df = pd.read_csv("city_level_data_0_0.csv")

# Data cleaning (your data cleaning code here...)
numeric_columns = [
    'composition_food_organic_waste_percent',
    'composition_glass_percent',
    'composition_metal_percent',
    'composition_other_percent',
    'composition_paper_cardboard_percent',
    'composition_plastic_percent',
    'composition_rubber_leather_percent',
    'composition_wood_percent',
    'composition_yard_garden_green_waste_percent',
    'total_msw_total_msw_generated_tons_year',
    'population_number_of_people',
    'n_waste_pickers_number_of_waste_pickers_number_of_people',
    'waste_collection_cost_recovery_household_fee_amount_na'
]

df[numeric_columns] = df[numeric_columns].replace(',', '', regex=True)
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df.fillna(df.mode().iloc[0], inplace=True)

# Handle special logic for zero population
df.loc[df['population_number_of_people'] == 0, 'total_msw_total_msw_generated_tons_year'] = 0


# Split the data into features and target for the first model
features_first_model = ['population_number_of_people',
                         'n_waste_pickers_number_of_waste_pickers_number_of_people',
                         'waste_collection_cost_recovery_household_fee_amount_na']

target_total_msw = 'total_msw_total_msw_generated_tons_year'

# Split the data for the first model
X_train_first_model, X_test_first_model, y_train_first_model, y_test_first_model = \
    train_test_split(df[features_first_model], df[target_total_msw], test_size=0.2, random_state=42)

# Train the first Decision Tree model
model_total_msw = DecisionTreeRegressor()
model_total_msw.fit(X_train_first_model, y_train_first_model)

# Make predictions and evaluate the first model
predictions_total_msw = model_total_msw.predict(X_test_first_model)
mae_total_msw = mean_absolute_error(y_test_first_model, predictions_total_msw)
r2_total_msw = r2_score(y_test_first_model, predictions_total_msw)

# Add the predictions to the dataset for the second model
df['predicted_total_msw'] = model_total_msw.predict(df[features_first_model])

# Split the data for the second model
features_second_model = ['population_number_of_people', 'predicted_total_msw']
target_composition = [
    'composition_food_organic_waste_percent',
    'composition_glass_percent',
    'composition_metal_percent',
    'composition_other_percent',
    'composition_paper_cardboard_percent',
    'composition_plastic_percent',
    'composition_rubber_leather_percent',
    'composition_wood_percent',
    'composition_yard_garden_green_waste_percent'
]
X_train_second_model, X_test_second_model, y_train_second_model, y_test_second_model = \
    train_test_split(df[features_second_model], df[target_composition], test_size=0.2, random_state=42)

# Train the second Decision Tree model based on the population and predicted total waste
model_composition = DecisionTreeRegressor()
model_composition.fit(X_train_second_model, y_train_second_model)

# Function to get display name for composition
def get_display_name(column_name):
    display_names = {
        'composition_food_organic_waste_percent': 'Food/Organic',
        'composition_glass_percent': 'Glass',
        'composition_metal_percent': 'Metal',
        'composition_other_percent': 'Other',
        'composition_paper_cardboard_percent': 'Paper/Cardboard',
        'composition_plastic_percent': 'Plastic',
        'composition_rubber_leather_percent': 'Rubber/Leather',
        'composition_wood_percent': 'Wood',
        'composition_yard_garden_green_waste_percent': 'Yard/Garden'
    }
    return display_names.get(column_name, column_name)

# Function to recommend strategy based on composition type
def recommend_strategy(composition_type):
    # Add specific recommendations based on the composition type
    if composition_type == 'composition_food_organic_waste_percent':
        st.write('**Recommendation:** Implement composting programs to manage organic waste effectively.')
        st.write('**Additional Strategy:** Educate the community on the benefits of composting.')
    elif composition_type == 'composition_glass_percent':
        st.write('**Recommendation:** Promote glass recycling and encourage the use of glass containers.')
        st.write('**Additional Strategy:** Educate the community on the benefits of glass recycling.')
    elif composition_type == 'composition_metal_percent':
        st.write('**Recommendation:** Establish metal recycling initiatives to recover valuable resources.')
        st.write('**Additional Strategy:** Collaborate with local businesses for metal recycling programs.')
    elif composition_type == 'composition_other_percent':
        st.write('**Recommendation:** Implement waste sorting programs to manage miscellaneous waste.')
        st.write('**Additional Strategy:** Promote awareness about proper waste disposal practices.')
    elif composition_type == 'composition_paper_cardboard_percent':
        st.write('**Recommendation:** Encourage paper and cardboard recycling to reduce environmental impact.')
        st.write('**Additional Strategy:** Promote the use of recycled paper and cardboard products.')
    elif composition_type == 'composition_plastic_percent':
        st.write('**Recommendation:** Implement a plastic recycling program to reduce environmental impact.')
        st.write('**Additional Strategy:** Encourage the use of biodegradable alternatives to reduce plastic usage.')
    elif composition_type == 'composition_rubber_leather_percent':
        st.write('**Recommendation:** Explore opportunities for recycling rubber and leather waste.')
        st.write('**Additional Strategy:** Educate the community on the importance of recycling rubber and leather.')
    elif composition_type == 'composition_wood_percent':
        st.write('**Recommendation:** Promote wood recycling and reuse programs.')
        st.write('**Additional Strategy:** Encourage the use of reclaimed wood for various purposes.')
    elif composition_type == 'composition_yard_garden_green_waste_percent':
        st.write('**Recommendation:** Implement composting programs for yard, garden, and green waste.')
        st.write('**Additional Strategy:** Promote community gardening and green waste recycling initiatives.')

# Streamlit UI with enhanced styling
st.title('Waste Prediction App')
st.markdown(
    """
    ## Predict and Visualize Waste Composition

    Adjust the sliders to input values and generate predictions.
    """
)

# Input sliders for the user to input values
population = st.slider('Population (Number of People)', min_value=1, max_value=100000, step=1)
waste_pickers = st.slider('Number of Waste Pickers', min_value=0, max_value=100000, step=1)
household_fee = st.slider('Household Fee Amount', min_value=0, max_value=1000, step=1)

# Button to generate prediction
if st.button('Generate Prediction'):
    # Check conditions for generating predictions
    if population == 0:
        st.warning("Population cannot be zero. Please adjust the input.")
    elif waste_pickers >= population:
        st.warning("The number of waste pickers cannot be greater than or equal to the population. Please adjust the input.")
    else:
        # Create a DataFrame with user input for the first model
        user_input_first_model = pd.DataFrame({
            'population_number_of_people': [population],
            'n_waste_pickers_number_of_waste_pickers_number_of_people': [waste_pickers],
            'waste_collection_cost_recovery_household_fee_amount_na': [household_fee]
        })

        # Generate predictions for the first model
        total_msw_prediction = model_total_msw.predict(user_input_first_model)

        # Add the prediction to the DataFrame for the second model
        user_input_second_model = pd.DataFrame({
            'population_number_of_people': [population],
            'predicted_total_msw': total_msw_prediction
        })

        # Generate predictions for the second model
        composition_prediction = model_composition.predict(user_input_second_model)

        # Display predictions with improved styling
        st.success(f'Predicted Total MSW: **{total_msw_prediction[0]:.2f} tons**')

        # Display composition bar graph with custom labels
        composition_df = pd.DataFrame(composition_prediction, columns=target_composition)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=composition_df, ax=ax, palette="viridis")

        # Customize x-axis labels
        ax.set_xticklabels(['Food/Organic', 'Glass', 'Metal', 'Other', 'Paper/Cardboard', 'Plastic', 'Rubber/Leather', 'Wood', 'Yard/Garden'], rotation=45, ha="right")

        plt.title('Predicted Composition Levels')
        plt.xlabel('Waste Types')
        plt.ylabel('Composition Percentage')
        st.pyplot(fig)

        # Strategy Recommendation based on highest composition
        highest_composition_type = target_composition[np.argmax(composition_prediction)]
        st.subheader('Strategy Recommendation')
        st.write(f'The highest expected composition is: **{get_display_name(highest_composition_type)}**')
        recommend_strategy(highest_composition_type)