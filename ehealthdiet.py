import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # Import ttk for styled widgets

# Load dataset
df = pd.read_csv("D:\project\e_health_diet_recommendations.csv")

# Data Preprocessing
df = df.dropna()

# Convert categorical variables to numeric
categorical_columns = ['Gender', 'Activity_Level', 'Dietary_Habits', 'Pre_Existing_Conditions', 'Medications']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Encode target variable
le = LabelEncoder()
df['Nutrient_Requirements'] = le.fit_transform(df['Nutrient_Requirements'])

# Feature selection
X = df.drop(columns=['Nutrient_Requirements', 'Meal_Plans'])
y = df['Nutrient_Requirements']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'diet_recommendation_model.pkl')

# Load the model
model = joblib.load('diet_recommendation_model.pkl')

# Diet recommendation function
def get_diet_recommendation(user_input):
    user_df = pd.DataFrame([user_input])
    
    # Apply preprocessing steps
    user_df = pd.get_dummies(user_df, columns=categorical_columns, drop_first=True)
    missing_cols = set(X.columns) - set(user_df.columns)
    for col in missing_cols:
        user_df[col] = 0
    user_df = user_df[X.columns]
    
    # Predict nutrient requirement
    user_pred = model.predict(user_df)
    nutrient_requirement = le.inverse_transform(user_pred)[0]
    
    # Ideal weight calculation (BMI range 18.5 - 24.9)
    height_m = user_input['Height'] / 100
    ideal_weight_min = 18.5 * (height_m ** 2)
    ideal_weight_max = 24.9 * (height_m ** 2)
    
    # BMI-based recommendations
    if user_input['BMI'] < 18.5:
        bmi_recommendation = "Increase calorie intake with a balanced diet to gain weight healthily."
    elif 18.5 <= user_input['BMI'] <= 24.9:
        bmi_recommendation = "Maintain your current diet and activity level to stay healthy."
    elif user_input['BMI'] > 24.9:
        bmi_recommendation = "Reduce calorie intake and increase physical activity to lose weight healthily."
    
    # Activity level-based recommendations
    if user_input['Activity_Level'] == 'Sedentary':
        activity_recommendation = "Consider incorporating more physical activity into your routine, like daily walks."
    elif user_input['Activity_Level'] == 'Lightly Active':
        activity_recommendation = "Try to increase activity levels with more regular exercise."
    elif user_input['Activity_Level'] == 'Active':
        activity_recommendation = "Maintain your current level of physical activity."
    elif user_input['Activity_Level'] == 'Very Active':
        activity_recommendation = "Ensure you're getting enough calories to support your activity level."
    
    # Map nutrient requirements to diet and food recommendations with levels
    diet_recommendations = {
        "High Protein": {
            "High": "Focus on foods like chicken breast, turkey, lean beef, tofu, Greek yogurt, and protein shakes.",
            "Medium": "Incorporate eggs, legumes, cottage cheese, and quinoa into your meals.",
            "Low": "Include nuts, seeds, and small amounts of lean meats in your diet."
        },
        "Low Carb": {
            "High": "Emphasize non-starchy vegetables, lean meats, and healthy fats like avocados and olive oil.",
            "Medium": "Include berries, yogurt, and low-carb vegetables like spinach and broccoli.",
            "Low": "Allow small amounts of whole grains and starchy vegetables."
        },
        "Low Fat": {
            "High": "Consume mostly fruits, vegetables, whole grains, and lean proteins like fish and chicken.",
            "Medium": "Add low-fat dairy products and avoid fried foods and high-fat meats.",
            "Low": "Include moderate amounts of healthy fats like olive oil and nuts."
        },
        "Balanced": {
            "High": "Ensure a mix of whole grains, lean proteins, healthy fats, and a variety of fruits and vegetables.",
            "Medium": "Focus on a balanced intake of macronutrients with an emphasis on variety.",
            "Low": "Maintain a general balance but allow flexibility with portion sizes."
        }
        # Add more mappings as needed
    }

    nutrient_level = "Medium"  # Determine level based on user input or other factors

    diet_recommendation = diet_recommendations.get(nutrient_requirement, {}).get(nutrient_level, "No specific recommendation available.")
    
    return nutrient_requirement, diet_recommendation, nutrient_level, (ideal_weight_min, ideal_weight_max), bmi_recommendation, activity_recommendation

# GUI Implementation with enhanced output
def submit():
    try:
        user_input = {
            'Age': int(entry_age.get()),
            'Height': int(entry_height.get()),
            'Weight': int(entry_weight.get()),
            'BMI': float(entry_bmi.get()),
            'Blood_Pressure': int(entry_bp.get()),
            'Cholesterol': int(entry_cholesterol.get()),
            'Blood_Sugar': int(entry_bs.get()),
            'Gender': var_gender.get(),
            'Activity_Level': var_activity.get(),
            'Dietary_Habits': var_dietary.get(),
            'Pre_Existing_Conditions': var_conditions.get(),
            'Medications': var_medications.get()
        }

        nutrient_requirement, diet_recommendation, nutrient_level, ideal_weight_range, bmi_recommendation, activity_recommendation = get_diet_recommendation(user_input)
        
        # Custom styled output message
        recommendation_message = (f"\n**Diet Recommendation**\n\n"
                                  f"**Nutrient Requirement**: {nutrient_requirement}\n"
                                  f"**Diet Level**: {nutrient_level}\n"
                                  f"**Diet Recommendation**: {diet_recommendation}\n\n"
                                  f"**Ideal Weight Range**: {ideal_weight_range[0]:.1f} kg - {ideal_weight_range[1]:.1f} kg\n\n"
                                  f"**BMI Recommendation**: {bmi_recommendation}\n\n"
                                  f"**Activity Recommendation**: {activity_recommendation}")
        
        # Display the recommendations in a styled pop-up
        styled_output(recommendation_message)
    
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")

def styled_output(message):
    output_window = tk.Toplevel(app)
    output_window.title("Diet Recommendation Result")
    output_window.configure(bg='#92cfd1')
    
    # Styled text widget to display output
    text_widget = tk.Text(output_window, wrap="word", bg="#f5f5f5", fg="#333333", font=("Arial", 12))
    text_widget.pack(padx=20, pady=20)
    
    # Insert the message and add custom styles
    text_widget.insert(tk.END, message)
    
    # Make text widget read-only
    text_widget.config(state=tk.DISABLED)

    # Close button
    ttk.Button(output_window, text="Close", command=output_window.destroy).pack(pady=10)

app = tk.Tk()
app.title("Diet Recommendation System")
app.configure(bg='#92cfd1')  # Background color

# Define styles
style = ttk.Style()
style.configure('TLabel', font=('Arial', 12), background='#f0f0f0', foreground='#333')
style.configure('TEntry', font=('Arial', 12), foreground='#333')
style.configure('TButton', font=('Arial', 12), background='Red', foreground='green')
style.configure('TOptionMenu', font=('Arial', 12), background='#f0f0f0', foreground='#333')

# Labels and inputs
ttk.Label(app, text="Age:").grid(row=0, padx=10, pady=5, sticky='W')
entry_age = ttk.Entry(app, width=20)
entry_age.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(app, text="Height (cm):").grid(row=1, padx=10, pady=5, sticky='W')
entry_height = ttk.Entry(app, width=20)
entry_height.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(app, text="Weight (kg):").grid(row=2, padx=10, pady=5, sticky='W')
entry_weight = ttk.Entry(app, width=20)
entry_weight.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(app, text="BMI:").grid(row=3, padx=10, pady=5, sticky='W')
entry_bmi = ttk.Entry(app, width=20)
entry_bmi.grid(row=3, column=1, padx=10, pady=5)

ttk.Label(app, text="Blood Pressure:").grid(row=4, padx=10, pady=5, sticky='W')
entry_bp = ttk.Entry(app, width=20)
entry_bp.grid(row=4, column=1, padx=10, pady=5)

ttk.Label(app, text="Cholesterol:").grid(row=5, padx=10, pady=5, sticky='W')
entry_cholesterol = ttk.Entry(app, width=20)
entry_cholesterol.grid(row=5, column=1, padx=10, pady=5)

ttk.Label(app, text="Blood Sugar:").grid(row=6, padx=10, pady=5, sticky='W')
entry_bs = ttk.Entry(app, width=20)
entry_bs.grid(row=6, column=1, padx=10, pady=5)

ttk.Label(app, text="Gender:").grid(row=7, padx=10, pady=5, sticky='W')
var_gender = tk.StringVar(value='Male')
gender_menu = ttk.OptionMenu(app, var_gender, 'Male', 'Male', 'Female')
gender_menu.grid(row=7, column=1, padx=10, pady=5)

ttk.Label(app, text="Activity Level:").grid(row=8, padx=10, pady=5, sticky='W')
var_activity = tk.StringVar(value='Sedentary')
activity_menu = ttk.OptionMenu(app, var_activity, 'Sedentary', 'Sedentary', 'Lightly Active', 'Active', 'Very Active')
activity_menu.grid(row=8, column=1, padx=10, pady=5)

ttk.Label(app, text="Dietary Habits:").grid(row=9, padx=10, pady=5, sticky='W')
var_dietary = tk.BooleanVar()
dietary_check = ttk.Checkbutton(app, variable=var_dietary)
dietary_check.grid(row=9, column=1, padx=10, pady=5)

ttk.Label(app, text="Pre-Existing Conditions:").grid(row=10, padx=10, pady=5, sticky='W')
var_conditions = tk.BooleanVar()
conditions_check = ttk.Checkbutton(app, variable=var_conditions)
conditions_check.grid(row=10, column=1, padx=10, pady=5)

ttk.Label(app, text="Medications:").grid(row=11, padx=10, pady=5, sticky='W')
var_medications = tk.BooleanVar()
medications_check = ttk.Checkbutton(app, variable=var_medications)
medications_check.grid(row=11, column=1, padx=10, pady=5)

# Submit button
submit_button = ttk.Button(app, text="Submit", command=submit)
submit_button.grid(row=12, columnspan=2, pady=10)

app.mainloop()
