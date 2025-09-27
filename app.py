import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
import pickle

# Set page configuration
st.set_page_config(
    page_title="House Price Analysis & Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class HousePriceAnalyzer:
    def __init__(self):
        self.df = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path=None, file_object=None):
        """Load data from file path or uploaded file object"""
        try:
            if file_object is not None:
                self.df = pd.read_csv(file_object)
            elif file_path is not None:
                self.df = pd.read_csv(file_path)
            else:
                st.error("Please provide either a file path or file object")
                return False
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def display_basic_info(self):
        """Display basic information about the dataset"""
        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Houses", f"{self.df.shape[0]:,}")
        
        with col2:
            st.metric("Number of Features", self.df.shape[1])
        
        with col3:
            missing_values = self.df.isnull().sum().sum()
            st.metric("Total Missing Values", missing_values)
        
        # Display dataframe
        st.subheader("Data Preview")
        st.dataframe(self.df.head(10))
        
        # Data types information
        st.subheader("Data Types Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            dtype_info = self.df.dtypes.reset_index()
            dtype_info.columns = ['Column', 'Data Type']
            st.dataframe(dtype_info)
        
        with col2:
            st.write("**Missing Values:**")
            missing_info = self.df.isnull().sum().reset_index()
            missing_info.columns = ['Column', 'Missing Values']
            st.dataframe(missing_info)
    
    def price_analysis(self):
        """Analyze price distribution and statistics"""
        st.markdown('<div class="section-header">💰 Price Analysis</div>', unsafe_allow_html=True)
        
        # Price statistics
        min_price = self.df["price"].min()
        max_price = self.df["price"].max()
        avg_price = self.df["price"].mean()
        median_price = self.df["price"].median()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Minimum Price", f"${min_price:,.0f}")
        with col2:
            st.metric("Maximum Price", f"${max_price:,.0f}")
        with col3:
            st.metric("Average Price", f"${avg_price:,.0f}")
        with col4:
            st.metric("Median Price", f"${median_price:,.0f}")
        
        # Price distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(self.df["price"], bins=50, edgecolor='black', alpha=0.7)
        ax1.set_title('Price Distribution')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(self.df["price"])
        ax2.set_title('Price Box Plot')
        ax2.set_ylabel('Price')
        
        st.pyplot(fig)
        
        # Remove extreme outliers for better visualization
        price_99 = np.percentile(self.df["price"], 99)
        filtered_prices = self.df[self.df["price"] <= price_99]["price"]
        
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.hist(filtered_prices, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_title('Price Distribution (99th Percentile)')
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig2)
    
    def city_analysis(self):
        """Analyze city-wise distribution"""
        st.markdown('<div class="section-header">🏙️ City Analysis</div>', unsafe_allow_html=True)
        
        # City with highest and lowest average prices
        avg_price_by_city = self.df.groupby("city")["price"].mean().sort_values(ascending=False)
        
        highest_avg_city = avg_price_by_city.idxmax()
        highest_avg_price = avg_price_by_city.max()
        
        lowest_avg_city = avg_price_by_city.idxmin()
        lowest_avg_price = avg_price_by_city.min()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("City with Highest Average Price", 
                     f"{highest_avg_city} (${highest_avg_price:,.0f})")
        
        with col2:
            st.metric("City with Lowest Average Price", 
                     f"{lowest_avg_city} (${lowest_avg_price:,.0f})")
        
        # Top 10 cities by average price
        st.subheader("Top 10 Cities by Average Price")
        top_10_cities = avg_price_by_city.head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        top_10_cities.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_title('Top 10 Cities by Average Price')
        ax.set_xlabel('City')
        ax.set_ylabel('Average Price ($)')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display full city list
        st.subheader("All Cities in Dataset")
        unique_cities = self.df["city"].unique()
        st.write(f"Total unique cities: {len(unique_cities)}")
        st.write(unique_cities)
    
    def bedroom_bathroom_analysis(self):
        """Analyze bedroom and bathroom distributions"""
        st.markdown('<div class="section-header">🛏️ Bedroom & Bathroom Analysis</div>', unsafe_allow_html=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bedroom count
        bedroom_counts = self.df['bedrooms'].value_counts().sort_index()
        axes[0].bar(bedroom_counts.index, bedroom_counts.values, color='lightblue', edgecolor='black')
        axes[0].set_title('Distribution of Bedrooms')
        axes[0].set_xlabel('Number of Bedrooms')
        axes[0].set_ylabel('Count')
        
        # Bathroom count
        bathroom_counts = self.df['bathrooms'].value_counts().sort_index()
        axes[1].bar(bathroom_counts.index, bathroom_counts.values, color='lightgreen', edgecolor='black')
        axes[1].set_title('Distribution of Bathrooms')
        axes[1].set_xlabel('Number of Bathrooms')
        axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Bedroom Statistics")
            avg_bedrooms = self.df['bedrooms'].mean()
            median_bedrooms = self.df['bedrooms'].median()
            st.metric("Average Bedrooms", f"{avg_bedrooms:.2f}")
            st.metric("Median Bedrooms", f"{median_bedrooms:.1f}")
        
        with col2:
            st.subheader("Bathroom Statistics")
            avg_bathrooms = self.df['bathrooms'].mean()
            median_bathrooms = self.df['bathrooms'].median()
            st.metric("Average Bathrooms", f"{avg_bathrooms:.2f}")
            st.metric("Median Bathrooms", f"{median_bathrooms:.2f}")
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        st.markdown('<div class="section-header">📈 Correlation Analysis</div>', unsafe_allow_html=True)
        
        # Select only numerical columns for correlation
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix Heatmap')
        st.pyplot(fig)
        
        # Top correlations with price
        price_correlations = correlation_matrix['price'].sort_values(ascending=False)
        st.subheader("Top Correlations with Price")
        
        corr_df = pd.DataFrame({
            'Feature': price_correlations.index,
            'Correlation with Price': price_correlations.values
        }).reset_index(drop=True)
        
        st.dataframe(corr_df.style.background_gradient(cmap='coolwarm', subset=['Correlation with Price']))
    
    def feature_distributions(self):
        """Show distributions of important features"""
        st.markdown('<div class="section-header">📋 Feature Distributions</div>', unsafe_allow_html=True)
        
        # Select important features to visualize
        important_features = ['sqft_living', 'sqft_lot', 'floors', 'yr_built', 'condition']
        
        for feature in important_features:
            if feature in self.df.columns:
                st.subheader(f"{feature.replace('_', ' ').title()} Distribution")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                self.df[feature].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
                ax.set_xlabel(feature.replace('_', ' ').title())
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Average {feature}", f"{self.df[feature].mean():.2f}")
                with col2:
                    st.metric(f"Median {feature}", f"{self.df[feature].median():.2f}")
                with col3:
                    st.metric(f"Std Dev {feature}", f"{self.df[feature].std():.2f}")
    
    def prepare_data_for_modeling(self):
        """Prepare data for machine learning models"""
        # For demonstration, let's create a classification problem: predict if price is above median
        median_price = self.df['price'].median()
        self.df['price_category'] = (self.df['price'] > median_price).astype(int)
        
        # Select features for modeling
        features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                   'condition', 'yr_built', 'sqft_above']
        
        X = self.df[features]
        y = self.df['price_category']
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple machine learning models"""
        st.markdown('<div class="section-header">🤖 Machine Learning Models</div>', unsafe_allow_html=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
        
        results = []
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            })
            
            # Store the trained model
            self.models[name] = model
        
        # Display results
        results_df = pd.DataFrame(results)
        st.subheader("Model Performance Comparison")
        
        # Format the results for better display
        styled_df = results_df.style.format({
            'Accuracy': '{:.3f}',
            'F1 Score': '{:.3f}',
            'ROC AUC': '{:.3f}'
        }).background_gradient(cmap='Blues', subset=['Accuracy', 'F1 Score', 'ROC AUC'])
        
        st.dataframe(styled_df)
        
        # Feature importance for Random Forest
        if 'Random Forest' in self.models:
            st.subheader("Random Forest Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.models['Random Forest'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
            ax.set_title('Feature Importance - Random Forest')
            st.pyplot(fig)
        
        return X_train, X_test, y_train, y_test

def main():
    st.markdown('<div class="main-header">🏠 House Price Analysis & Prediction Dashboard</div>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = HousePriceAnalyzer()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    sections = [
        "Data Overview",
        "Price Analysis", 
        "City Analysis",
        "Bedroom & Bathroom Analysis",
        "Correlation Analysis",
        "Feature Distributions",
        "Machine Learning Models"
    ]
    selected_section = st.sidebar.selectbox("Choose a section:", sections)
    
    # File upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    # Use sample data if no file uploaded
    if uploaded_file is not None:
        if analyzer.load_data(file_object=uploaded_file):
            st.sidebar.success("Data loaded successfully!")
    else:
        st.sidebar.info("Using sample data from the notebook")
        # For demo purposes, we'll create sample data similar to the notebook structure
        sample_data = {
            'date': pd.date_range('2014-05-01', periods=100, freq='D'),
            'price': np.random.normal(500000, 200000, 100),
            'bedrooms': np.random.randint(1, 6, 100),
            'bathrooms': np.random.uniform(1, 3, 100),
            'sqft_living': np.random.randint(1000, 3000, 100),
            'sqft_lot': np.random.randint(5000, 15000, 100),
            'floors': np.random.choice([1, 1.5, 2, 2.5, 3], 100),
            'waterfront': np.random.randint(0, 2, 100),
            'view': np.random.randint(0, 5, 100),
            'condition': np.random.randint(1, 6, 100),
            'sqft_above': np.random.randint(500, 2500, 100),
            'sqft_basement': np.random.randint(0, 1000, 100),
            'yr_built': np.random.randint(1950, 2015, 100),
            'yr_renovated': np.random.randint(0, 2015, 100),
            'city': np.random.choice(['Seattle', 'Bellevue', 'Redmond', 'Kent', 'Shoreline'], 100)
        }
        analyzer.df = pd.DataFrame(sample_data)
    
    if analyzer.df is not None:
        # Display selected section
        if selected_section == "Data Overview":
            analyzer.display_basic_info()
            
        elif selected_section == "Price Analysis":
            analyzer.price_analysis()
            
        elif selected_section == "City Analysis":
            analyzer.city_analysis()
            
        elif selected_section == "Bedroom & Bathroom Analysis":
            analyzer.bedroom_bathroom_analysis()
            
        elif selected_section == "Correlation Analysis":
            analyzer.correlation_analysis()
            
        elif selected_section == "Feature Distributions":
            analyzer.feature_distributions()
            
        elif selected_section == "Machine Learning Models":
            try:
                X, y = analyzer.prepare_data_for_modeling()
                analyzer.train_models(X, y)
                
                # Price prediction interface
                st.sidebar.markdown("---")
                st.sidebar.subheader("Price Category Prediction")
                
                st.sidebar.write("Enter house features to predict if price is above median:")
                
                bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
                bathrooms = st.sidebar.slider("Bathrooms", 1.0, 4.0, 2.0)
                sqft_living = st.sidebar.slider("Square Feet Living", 500, 4000, 1500)
                sqft_lot = st.sidebar.slider("Square Feet Lot", 1000, 20000, 8000)
                floors = st.sidebar.slider("Floors", 1, 3, 2)
                condition = st.sidebar.slider("Condition (1-5)", 1, 5, 3)
                yr_built = st.sidebar.slider("Year Built", 1900, 2020, 1985)
                sqft_above = st.sidebar.slider("Square Feet Above", 500, 3000, 1500)
                
                if st.sidebar.button("Predict Price Category"):
                    # Prepare input features
                    input_features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, 
                                              floors, condition, yr_built, sqft_above]])
                    
                    # Scale features
                    input_scaled = analyzer.scaler.transform(input_features)
                    
                    # Get prediction from Random Forest
                    if 'Random Forest' in analyzer.models:
                        prediction = analyzer.models['Random Forest'].predict(input_scaled)[0]
                        probability = analyzer.models['Random Forest'].predict_proba(input_scaled)[0][1]
                        
                        median_price = analyzer.df['price'].median()
                        
                        if prediction == 1:
                            st.sidebar.success(f"Prediction: Above Median Price (${median_price:,.0f}+)")
                        else:
                            st.sidebar.info(f"Prediction: Below Median Price (${median_price:,.0f}-)")
                        
                        st.sidebar.write(f"Probability of being above median: {probability:.2%}")
                        
            except Exception as e:
                st.error(f"Error in machine learning section: {str(e)}")
                st.info("This might be due to missing features in the dataset. Please ensure your dataset has the required columns.")
    
    else:
        st.info("Please upload a CSV file to get started with the analysis.")

if __name__ == "__main__":
    main()