# imports
import pandas as pd
import numpy as np
import os
import re
import itertools #NI Added
import spacy
import emoji
from PIL import Image
import torch
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D, Concatenate, Input
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import chardet


data_dir = './Dataset/'

class Loader():
    """Loads the dataset from the directory."""
    data_dir = ""
    image_dir = ""
    annotations_dir = ""
    image_dir = ""
    limit = None
    anotations_dfs = []
    image_paths = {}
    skipped_files = [] #NI for skipped files

    def __init__(self, data_dir, 
                 annotations_dir = "annotations", 
                 image_dir = "data_image", 
                 limit=None):
        """Initializes the Loader with the dataset directory.
        Args:
            data_dir: Path to the dataset directory
            annotations_dir: Directory containing the annotation files
            image_dir: Directory containing the image files
            limit: Maximum number of samples to load from each file (for testing)
        """
        self.data_dir = data_dir
        self.annotations_dir = os.path.join(self.data_dir, annotations_dir)
        self.image_dir = os.path.join(self.data_dir,image_dir)
        self.limit = limit

    def load_from_file(self, filename):
        """Loads an annotation file from the dataset directory.
        Args:
            filename: Name of the annotation file
        Returns:
            DataFrame containing the annotation data
        """
        if filename.endswith(".tsv"):
            filepath = os.path.join(self.annotations_dir, filename)

            # Detect the file's encoding
            with open(filepath, 'rb') as f:
                result = chardet.detect(f.read())
            file_encoding = result['encoding']
            try:
                # Try reading with the detected encoding
                df = pd.read_csv(filepath, sep='\t', encoding=file_encoding)
            except UnicodeDecodeError:
                # If the file_encoding failed, default to 'latin-1'
                try:
                    df = pd.read_csv(filepath, sep='\t', encoding='latin-1')
                except Exception as e:
                    print(f"Error reading file {filename} with encoding latin-1 or {file_encoding}: {e}")
                    self.skipped_files.append(filename)
                    return None # skip to the next file
            if self.limit != None:
                df = df.head(self.limit)
            for index, row in df.iterrows():
                #tweet_id = row['tweet_id']
                self.image_paths[row['image_id']] = self.data_dir + row['image_path']
            self.anotations_dfs.append(df)
            return df
        return None
    
    def load_files(self):
        """Loads all annotation files from the dataset directory.
        Returns:
            DataFrame containing the combined annotation data
        """
        for filename in os.listdir(self.annotations_dir):
            self.load_from_file(filename)
        return pd.concat(self.anotations_dfs, ignore_index=True)

class TextPreprocessor():
    nlp = None
    df = None
    def __init__(self, df):
        self.df = df
        self.nlp = spacy.load("en_core_web_sm") # Load spaCy model

    def preprocess_text(self, text):
        """Cleans and preprocesses the tweet text."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text) # Remove special characters
        text = text.lower().strip() # Lowercase and strip whitespace
        text = re.sub(r'RT\s@\w+:', '', text) # Remove retweets
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = emoji.demojize(text) # convert emojis to text
        return text

    def extract_location(self, text):
        """Extracts location entities from the tweet text."""
        doc = self.nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        return locations

    def prepare_tweets(self):
        # Remove empty tweets
        self.df = self.df[~self.df['tweet_text'].str.len().lt(1)]
        # Remove retweets
        self.df = self.df[~self.df['tweet_text'].str.startswith('RT @', na=False)]
        # Work with spacy
        self.df['processed_text'] = self.df['tweet_text'].apply(self.preprocess_text)
        self.df['locations'] = self.df['processed_text'].apply(lambda x: self.extract_location(x))
        return None

    def prepare_labels(self):
        self.df.loc[:,['informative']] = self.df['text_info'].apply(lambda x: 1 if x == 'informative' else 0)
        self.df.loc[:,['damage_severity']] = self.df['image_damage'].map({'severe_damage': 2, 
                                                        'mild_damage': 1, 
                                                        'little_or_no_damage': 0})
        return None
    
    def preprocess_tweets(self):
        """Pipeline for Preprocesses the tweet text and extracts location entities."""
        self.prepare_tweets()
        self.prepare_labels()
        return self.df
    # --- Sentiment Analysis ---

    def analyze_sentiment(self,text):
        """Analyzes sentiment of the tweet text using VADER."""
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(text)
        return vs['compound']

    def extract_sentiment_scores(self):
        """Extracts sentiment scores for each tweet."""
        self.df.loc[:,['sentiment_score']] = self.df['processed_text'].apply(self.analyze_sentiment)
        return self.df

class ImgPreprocessor():
    df = None
    image_paths = None
    transform = None
    def __init__(self, df, image_paths):
        self.image_paths = image_paths
        self.tranform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.df = df
    
    def preprocess_image(self, image_path):
        """Preprocesses the image."""
        try:
            img = Image.open(image_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def create_image_tensors(self):
        """Creates image tensors from the image paths."""
        self.df['image_tensor'] = self.df['image_id'].apply(lambda x: self.preprocess_image(self.image_paths[x]))
        self.df = self.df.dropna(subset=['image_tensor'])
        return None
    
    def get_preprocessed_images(self):
        self.create_image_tensors()
        return self.df
    
class Preprocessor():
    """Preprocesses the data for training."""
    df = None
    image_paths = None
    def __init__(self,df, image_paths):
        self.df = df
        self.image_paths = image_paths
    
    def preprocess_data(self):
        text_preprocessor = TextPreprocessor(self.df)
        self.df = text_preprocessor.preprocess_tweets()
        img_preprocessor = ImgPreprocessor(self.df, self.image_paths)
        self.df = img_preprocessor.get_preprocessed_images()
        return self.df

class Helper():
    @staticmethod
    def print_class_imbalance(df):
        """Prints the class imbalance in the dataset."""
        informative_counts = df['informative'].value_counts()
        damage_counts = df['damage_severity'].value_counts()
        print("Informative Label Counts:\n", informative_counts)
        print("\nDamage Severity Label Counts:\n", damage_counts)
        return None
    
    @staticmethod
    def print_class_imbalance(df):
        # Handle class imbalance (basic example - you can explore more advanced techniques)
        informative_counts = df['informative'].value_counts()
        damage_counts = df['damage_severity'].value_counts()
        # Print the counts to inspect the balance.
        print("Informative Label Counts:\n", informative_counts)
        print("\nDamage Severity Label Counts:\n", damage_counts)
        return None
    
    @staticmethod
    def split_data(df):
        """Splits the preprocessed data into training, validation, and test sets."""
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        return train_df, val_df, test_df

    @staticmethod
    def print_classification_report(msg, y_val, predictions):
        print(msg)
        print(classification_report(y_val, predictions, zero_division=0))
        return None

class Naive():
    X_train = None
    y_train = None
    X_val = None
    y_val = None

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def train_naive(self):
        """Naive Bayes model training."""
        nb_model = MultinomialNB()
        nb_model.fit(self.X_train, self.y_train)
        nb_predictions = nb_model.predict(self.X_val)
        return nb_model, nb_predictions
    
    def prn_classification(self):
        Helper.print_classification_report("Naive Bayes Classification Report:", 
                                           self.y_val, self.nb_predictions)
    
class SVM():
    X_train = None
    y_train = None
    X_val = None
    y_val = None

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def train_svm(self):
        """SVM model training."""
        svm_model = SVC()
        svm_model.fit(self.X_train, self.y_train)
        svm_predictions = svm_model.predict(self.X_val)
        return svm_model, svm_predictions

    def prn_classification(self):
        Helper.print_classification_report("SVM Classification Report:", 
                                           self.y_val, self.svm_predictions)
    
class RandomForest():
    X_train = None
    y_train = None
    X_val = None
    y_val = None

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def train_random_forest(self):
        """Random Forest model training."""
        rf_model = RandomForestClassifier()
        rf_model.fit(self.X_train, self.y_train)
        rf_predictions = rf_model.predict(self.X_val)
        return rf_model, rf_predictions
    
    def prn_classification(self):
        Helper.print_classification_report("Random Forest Classification Report:", 
                                           self.y_val, self.rf_predictions)
    
class LSTM():
    X_train = None
    y_train = None
    X_val = None
    y_val = None

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
    
    def train_lstm(self):
        """LSTM model training."""
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(self.X_train)
        X_train_seq = tokenizer.texts_to_sequences(self.X_train)
        X_val_seq = tokenizer.texts_to_sequences(self.X_val)
        vocab_size = len(tokenizer.word_index) + 1
        embedding_dim = 200
        max_len = max([len(x) for x in X_train_seq])
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
        X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
        lstm_model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim),
            LSTM(128),
            Dense(1, activation='sigmoid')
        ])
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm_model.fit(X_train_pad, self.y_train, validation_data=(X_val_pad, self.y_val), epochs=5, batch_size=32)
        lstm_predictions = (lstm_model.predict(X_val_pad) > 0.5).astype("int32")
        return lstm_model, lstm_predictions
    
    def prn_classification(self):
        Helper.print_classification_report("LSTM Classification Report:", 
                                           self.y_val, self.lstm_predictions, 
                                           zero_division=0)
        
class TextTrain():
    def vectorization(train_df, val_df, test_df):
        """TF-IDF Vectorization"""
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['processed_text'])
        X_val_tfidf = tfidf_vectorizer.transform(val_df['processed_text'])
        X_test_tfidf = tfidf_vectorizer.transform(test_df['processed_text'])
        return X_train_tfidf, X_val_tfidf, X_test_tfidf

    def label_encoding(train_df, val_df, test_df):
        """Label Encoding"""
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_df['informative'])
        y_val = label_encoder.transform(val_df['informative'])
        y_test = label_encoder.transform(test_df['informative'])
        return y_train, y_val, y_test
    
class ImageTrain():
    train_df = None
    val_df = None
    def __init__(self, train_df, val_df):
        self.train_df = train_df
        self.val_df = val_df

    def extract_image_features(self, image_tensors, model):
        """Extracts features from image tensors using a pre-trained model
        Args:
            image_tensors: List of image tensors
            model: Pre-trained image model  
        Returns:
            List of image features
        """
        model.eval()
        features = []
        with torch.no_grad():
            for tensor in image_tensors:
                if tensor is not None:
                    tensor = tensor.unsqueeze(0)
                    feature = model(tensor)
                    features.append(feature.squeeze().numpy())
                else:
                    features.append(None)
        return features

    def get_resnet_model(self):
        resnet_model = models.resnet50(pretrained=True)
        resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
        return resnet_model
    
    def train_resnet_model(self):
        # Extract image features
        resnet_model = self.get_resnet_model()
        train_resnet_features = self.extract_image_features(self.train_df['image_tensor'].tolist(), resnet_model)
        val_resnet_features = self.extract_image_features(self.val_df['image_tensor'].tolist(), resnet_model)
        
        self.train_df['resnet_features'] = train_resnet_features
        self.train_df = self.train_df.dropna(subset=['resnet_features'])
        self.val_df['resnet_features'] = val_resnet_features
        self.val_df = self.val_df.dropna(subset=['resnet_features'])
        return self.train_df, self.val_df

    def get_vgg_model(self):
        vgg_model = models.vgg16(pretrained=True)
        vgg_model = torch.nn.Sequential(*(list(vgg_model.features)))
        return vgg_model
    
    def train_vgg_model(self):
        # Extract image features
        vgg_model = self.get_vgg_model()
        train_vgg_features = self.extract_image_features(self.train_df['image_tensor'].tolist(), vgg_model)
        val_vgg_features = self.extract_image_features(self.val_df['image_tensor'].tolist(), vgg_model)

        self.train_df['vgg_features'] = train_vgg_features
        self.train_df = self.train_df.dropna(subset=['vgg_features'])
        self.val_df['vgg_features'] = val_vgg_features
        self.val_df = self.val_df.dropna(subset=['vgg_features'])
        
        return self.train_df, self.val_df
    
    def get_efficientnet_model(self):
        efficientnet_model = models.efficientnet_b0(pretrained=True)
        efficientnet_model = torch.nn.Sequential(*(list(efficientnet_model.children())[:-1]))
        return efficientnet_model
    
    def train_efficientnet_model(self):
        # Extract image features
        efficientnet_model = self.get_efficientnet_model()
        train_efficientnet_features = self.extract_image_features(self.train_df['image_tensor'].tolist(), efficientnet_model)
        val_efficientnet_features = self.extract_image_features(self.val_df['image_tensor'].tolist(), efficientnet_model)

        self.train_dftrain_df['efficientnet_features'] = train_efficientnet_features
        self.train_df = self.train_df.dropna(subset=['efficientnet_features'])
        self.val_df['efficientnet_features'] = val_efficientnet_features
        self.val_df = self.val_df.dropna(subset=['efficientnet_features'])
        return self.train_df, self.val_df

def train_image_models(train_df, val_df):
    # Label Encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['informative'])
    y_val = label_encoder.transform(val_df['informative'])

    # Train and evaluate classifiers (e.g., Random Forest)
    resnet_rf = RandomForestClassifier()
    resnet_rf.fit(np.array(train_df['resnet_features'].tolist()), y_train)
    resnet_predictions = resnet_rf.predict(np.array(val_df['resnet_features'].tolist()))
    print("ResNet Random Forest Classification Report:")
    print(classification_report(y_val, resnet_predictions, zero_division=0))

    vgg_rf = RandomForestClassifier()
    #NI ERROR Το πρόβλημα δημιουργείται εδώ. Το vgg_features επιστρέφει 4dim
    tt = np.array(train_df['vgg_features'].tolist()) #NI Remove
    print(f"vgg dimensions {tt.ndim}, vgg shape {tt.shape}") #NI Remove
    vgg_rf.fit(np.array(train_df['vgg_features'].tolist()), y_train)
    vgg_predictions = vgg_rf.predict(np.array(val_df['vgg_features'].tolist()))
    print("VGG Random Forest Classification Report:")
    print(classification_report(y_val, vgg_predictions, zero_division=0))

    efficientnet_rf = RandomForestClassifier()
    efficientnet_rf.fit(np.array(train_df['efficientnet_features'].tolist()), y_train)
    efficientnet_predictions = efficientnet_rf.predict(np.array(val_df['efficientnet_features'].tolist()))
    print("EfficientNet Random Forest Classification Report:")
    print(classification_report(y_val, efficientnet_predictions, zero_division=0))

    return resnet_model, vgg_model, efficientnet_model, resnet_rf, vgg_rf, efficientnet_rf, label_encoder




# --- Multimodal Fusion ---

def create_multimodal_model(text_features_shape, image_features_shape, num_classes):
    """Creates a multimodal model with late fusion."""
    text_input = Input(shape=(text_features_shape,))
    image_input = Input(shape=(image_features_shape,))

    concat_features = Concatenate()([text_input, image_input])
    dense1 = Dense(128, activation='relu')(concat_features)
    output = Dense(num_classes, activation='softmax')(dense1)

    model = Model(inputs=[text_input, image_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def fuse_and_train_multimodal(train_df, val_df, test_df, nb_model, svm_model, rf_model, resnet_rf, vgg_rf, efficientnet_rf, tfidf_vectorizer, label_encoder_text):
    """Fuses text and image features and trains a multimodal model."""

    # Text Features (TF-IDF)
    X_train_text = tfidf_vectorizer.transform(train_df['processed_text']).toarray()
    X_val_text = tfidf_vectorizer.transform(val_df['processed_text']).toarray()
    X_test_text = tfidf_vectorizer.transform(test_df['processed_text']).toarray()

    # Image Features (Random Forest features)
    X_train_resnet = np.array(train_df['resnet_features'].tolist())
    X_val_resnet = np.array(val_df['resnet_features'].tolist())
    X_test_resnet = np.array(test_df['resnet_features'].tolist())

    X_train_vgg = np.array(train_df['vgg_features'].tolist())
    X_val_vgg = np.array(val_df['vgg_features'].tolist())
    X_test_vgg = np.array(test_df['vgg_features'].tolist())

    X_train_efficient = np.array(train_df['efficientnet_features'].tolist())
    X_val_efficient = np.array(val_df['efficientnet_features'].tolist())
    X_test_efficient = np.array(test_df['efficientnet_features'].tolist())

    # Sentiment Features
    X_train_sentiment = np.array(train_df['sentiment_score']).reshape(-1, 1)
    X_val_sentiment = np.array(val_df['sentiment_score']).reshape(-1, 1)
    X_test_sentiment = np.array(test_df['sentiment_score']).reshape(-1, 1)

    # Location Features
    X_train_locations = train_df['locations'].apply(lambda x: 1 if len(x) > 0 else 0).values.reshape(-1, 1)
    X_val_locations = val_df['locations'].apply(lambda x: 1 if len(x) > 0 else 0).values.reshape(-1, 1)
    X_test_locations = test_df['locations'].apply(lambda x: 1 if len(x) > 0 else 0).values.reshape(-1, 1)

    # Concatenate features
    X_train_combined = np.concatenate([X_train_text, X_train_resnet, X_train_vgg, X_train_efficient, X_train_sentiment, X_train_locations], axis=1)
    X_val_combined = np.concatenate([X_val_text, X_val_resnet, X_val_vgg, X_val_efficient, X_val_sentiment, X_val_locations], axis=1)
    X_test_combined = np.concatenate([X_test_text, X_test_resnet, X_test_vgg, X_test_efficient, X_test_sentiment, X_test_locations], axis=1)

    # Label Encoding
    y_train = label_encoder_text.transform(train_df['informative'])
    y_val = label_encoder_text.transform(val_df['informative'])
    y_test = label_encoder_text.transform(test_df['informative'])

    # Create and train multimodal model
    multimodal_model = create_multimodal_model(X_train_combined.shape[1], 0, len(label_encoder_text.classes_))
    multimodal_model.fit(X_train_combined, y_train, validation_data=(X_val_combined, y_val), epochs=10, batch_size=32)

    # Evaluate multimodal model
    multimodal_predictions = np.argmax(multimodal_model.predict(X_test_combined), axis=1)
    print("Multimodal Model Classification Report:")
    print(classification_report(y_test, multimodal_predictions, zero_division=0))

    return multimodal_model

# --- Model Evaluation and Analysis ---

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluates the model and prints performance metrics."""
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0, target_names=label_encoder.classes_))

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def evaluate_sentiment(df):
    """Evaluates sentiment analysis performance."""
    # Placeholder for sentiment evaluation (e.g., compare with human-annotated sentiment)
    print("Sentiment Analysis Evaluation (Placeholder):")
    print("This section requires human-annotated sentiment labels for proper evaluation.")

def error_analysis(model, X_test, y_test, label_encoder, test_df):
    """Performs error analysis and visualizes the confusion matrix."""
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Analyze Misclassified Samples
    misclassified_indices = np.where(y_pred != y_test)[0]
    misclassified_samples = test_df.iloc[misclassified_indices]
    print("\nMisclassified Samples:")
    print(misclassified_samples[['tweet_text', 'informative']].head()) # Display the tweet text and the correct informative label.

# --- Model Evaluation and Analysis ---

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluates the model and prints performance metrics."""
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0, target_names=label_encoder.classes_))

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def evaluate_sentiment(df):
    """Evaluates sentiment analysis performance."""
    # Placeholder for sentiment evaluation (e.g., compare with human-annotated sentiment)
    print("Sentiment Analysis Evaluation (Placeholder):")
    print("This section requires human-annotated sentiment labels for proper evaluation.")

def error_analysis(model, X_test, y_test, label_encoder, test_df):
    """Performs error analysis and visualizes the confusion matrix."""
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Analyze Misclassified Samples
    misclassified_indices = np.where(y_pred != y_test)[0]
    misclassified_samples = test_df.iloc[misclassified_indices]
    print("\nMisclassified Samples:")
    print(misclassified_samples[['tweet_text', 'informative']].head()) # Display the tweet text and the correct informative label.


def main():
    # Running the model
    preprocessed_df = prepare_data(data_dir)
    preprocessed_df.head()
    train_df, val_df, test_df = split_data(preprocessed_df)
    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)
    preprocessed_df = extract_sentiment_scores(preprocessed_df)
    train_df, val_df, test_df = split_data(preprocessed_df)
    nb_model, svm_model, rf_model, lstm_model, tfidf_vectorizer, tokenizer, max_len, label_encoder_text = train_text_models(train_df, val_df, test_df)
    resnet_model, vgg_model, efficientnet_model, resnet_rf, vgg_rf, efficientnet_rf, label_encoder_image = train_image_models(train_df, val_df)
    multimodal_model = fuse_and_train_multimodal(train_df, val_df, test_df, nb_model, svm_model, rf_model, resnet_rf, vgg_rf, efficientnet_rf, tfidf_vectorizer, label_encoder_text)

    # Prepare test data for evaluation
    X_test_text = tfidf_vectorizer.transform(test_df['processed_text']).toarray()
    X_test_resnet = np.array(test_df['resnet_features'].tolist())
    X_test_vgg = np.array(test_df['vgg_features'].tolist())
    X_test_efficient = np.array(test_df['efficientnet_features'].tolist())
    X_test_sentiment = np.array(test_df['sentiment_score']).reshape(-1, 1)
    X_test_locations = test_df['locations'].apply(lambda x: 1 if len(x) > 0 else 0).values.reshape(-1, 1)
    X_test_combined = np.concatenate([X_test_text, X_test_resnet, X_test_vgg, X_test_efficient, X_test_sentiment, X_test_locations], axis=1)
    y_test = label_encoder_text.transform(test_df['informative'])

    # Evaluate Multimodal Model
    evaluate_model(multimodal_model, X_test_combined, y_test, label_encoder_text)

    # Evaluate Sentiment Analysis
    evaluate_sentiment(test_df)

    # Error Analysis
    error_analysis(multimodal_model, X_test_combined, y_test, label_encoder_text, test_df)

    # Visualize Results
    y_pred_probs = multimodal_model.predict(X_test_combined)
    y_pred = np.argmax(y_pred_probs, axis=1)
    # visualize_results(test_df, y_test, y_pred, label_encoder_text)

if __name__ == "__main__":
    loader = Loader("./Dataset/", limit=10)
    #df = loader.load_from_file("california_wildfires_final_data.tsv")
    df = loader.load_files()
    text_preprocessor = TextPreprocessor(df)
    df = text_preprocessor.preprocess_tweets()
    print(df.shape)
    print(df.head(10))