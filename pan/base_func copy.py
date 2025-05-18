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
        """Initializes the Loader with the dataset directory."""
        self.data_dir = data_dir
        self.annotations_dir = os.path.join(self.data_dir, annotations_dir)
        self.image_dir = os.path.join(self.data_dir,image_dir)
        self.limit = limit

    def load_from_file(self, filename):
        """Loads an annotation file from the dataset directory."""
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
        """Loads all annotation files from the dataset directory."""
        for filename in os.listdir(self.annotations_dir):
            self.load_from_file(filename)
        return pd.concat(self.anotations_dfs, ignore_index=True)

    

def load_crisis_mmd_dataset(data_dir): #NI Changed
    #NI The problem here is that creates two unnamed columns at the end of the dataframe. 
    annotations_dfs = []
    image_paths = {}
    skipped_files = [] #NI for skipped files

    annotations_dir = os.path.join(data_dir, "annotations")
    image_dir = os.path.join(data_dir, "data_image") #NI Unused Variable

    # Iterate through each disaster event's annotation file
    for filename in os.listdir(annotations_dir):
        if filename.endswith(".tsv"):
            filepath = os.path.join(annotations_dir, filename)

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
                skipped_files.append(filename) #NI for skipped files
                continue # skip to the next file

            annotations_dfs.append(df)

            # Create image path mappings
            for index, row in df.iterrows():
                tweet_id = row['tweet_id']
                image_paths[row['image_id']] = data_dir + row['image_path']

    # Concatenate all annotation DataFrames
    annotations_df = pd.concat(annotations_dfs, ignore_index=True)

    return annotations_df, image_paths

### Text Preprocessing Functions ###
def preprocess_text(text):#NI Changed
    """Cleans and preprocesses the tweet text."""
    #NI Συνειδητά αφαιρώ όλα τα κενά? Αν ναι, γιατί;
    if not isinstance(text, str):
        return "" # Handle non-string inputs
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text) # Remove special characters
    text = text.lower().strip() # Lowercase and strip whitespace
    text = re.sub(r'RT\s@\w+:', '', text) # Remove retweets
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = emoji.demojize(text) # convert emojis to text
    return text

def extract_location(text, nlp):
    """Extracts location entities from the tweet text."""
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations

def prepare_tweets(df):
    # Remove retweets
    df = df[~df['tweet_text'].str.startswith('RT @', na=False)]
    # Work with spacy
    nlp = spacy.load("en_core_web_sm") # Load spaCy model
    df['processed_text'] = df['tweet_text'].apply(preprocess_text)
    df['locations'] = df['processed_text'].apply(lambda x: extract_location(x, nlp))
    return df

def prepare_labels(df):
    df['informative'] = df['text_info'].apply(lambda x: 1 if x == 'informative' else 0)
    df['damage_severity'] = df['image_damage'].map({'severe_damage': 2, 
                                                    'mild_damage': 1, 
                                                    'little_or_no_damage': 0})
    return df

### End of Text Preprocessing Functions ###

### Image Preprocessing Functions ###
def preprocess_image(image_path, transform):
    """Preprocesses the image."""
    try:
        img = Image.open(image_path).convert('RGB')
        return transform(img)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def create_image_tensors(df, image_paths):
    """Creates image tensors from the image paths."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df['image_tensor'] = df['image_id'].apply(lambda x: preprocess_image(image_paths[x], transform))
    df = df.dropna(subset=['image_tensor']) # remove rows where image loading failed.
    return df

### End of Image Preprocessing Functions ###

### Evaluation Helper Functions ###
def print_class_imbalance(df):
    # Handle class imbalance (basic example - you can explore more advanced techniques)
    informative_counts = df['informative'].value_counts()
    damage_counts = df['damage_severity'].value_counts()

    # Print the counts to inspect the balance.
    print("Informative Label Counts:\n", informative_counts)
    print("\nDamage Severity Label Counts:\n", damage_counts)

### End Evaluation Helper Functions ###
def prepare_data(data_dir):
    """Prepares the dataset for model training."""
    # Load anotations and image_paths
    annotations_df, image_paths = load_crisis_mmd_dataset(data_dir)
    
    # Use part of the dataset
    annotations_df = annotations_df.head(100)
    
    # Prepare retweets
    annotations_df = prepare_tweets(annotations_df)

    # Create image tensors and remove rows with missing image tensors
    annotations_df = create_image_tensors(annotations_df, image_paths)

    # Label preparation
    annotations_df = prepare_labels(annotations_df)
    
    # Print class imbalance
    print_class_imbalance(annotations_df)

    return annotations_df

def split_data(preprocessed_df):
    """Splits the preprocessed data into training, validation, and test sets."""
    train_df, temp_df = train_test_split(preprocessed_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df

### Text Models Selection and Training ###
def print_classification_report(msg, y_val, predictions):
    print(msg)
    print(classification_report(y_val, predictions, zero_division=0))

def train_naive(X_train, y_train, X_val, y_val):
    """Naive Bayes model training."""
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_predictions = nb_model.predict(X_val)
    return nb_model, nb_predictions

def train_svn(X_train, y_train, X_val, y_val):
    """SVM model training."""
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_val)
    return svm_model, svm_predictions

def train_random_forest(X_train, y_train, X_val, y_val):
    """Random Forest model training."""
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_val)
    return rf_model, rf_predictions

### End of Text Models Selection and Training ###
def train_text_models(train_df, val_df, test_df):
    """Trains and evaluates different text models."""

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['processed_text'])
    X_val_tfidf = tfidf_vectorizer.transform(val_df['processed_text'])
    X_test_tfidf = tfidf_vectorizer.transform(test_df['processed_text'])

    # Label Encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['informative'])
    y_val = label_encoder.transform(val_df['informative'])
    y_test = label_encoder.transform(test_df['informative'])

    # Naive Bayes
    nb_model, nb_predictions = train_naive(X_train_tfidf, y_train, X_val_tfidf, y_val)
    print_classification_report("Naive Bayes Classification Report:", y_val, nb_predictions)

    # SVM
    svm_model, svm_predictions =  train_svn(X_train_tfidf, y_train, X_val_tfidf, y_val)
    print_classification_report("SVM Classification Report:", y_val, svm_predictions)

    # Random Forest
    rf_model, rf_predictions = train_random_forest(X_train_tfidf, y_train, X_val_tfidf, y_val)
    print_classification_report("Random Forest Classification Report:", y_val, rf_predictions)

    # LSTM Model (Deep Learning)
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_df['processed_text'])
    X_train_seq = tokenizer.texts_to_sequences(train_df['processed_text'])
    X_val_seq = tokenizer.texts_to_sequences(val_df['processed_text'])
    X_test_seq = tokenizer.texts_to_sequences(test_df['processed_text'])

    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 200

    max_len = max([len(x) for x in X_train_seq])
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    lstm_model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val), epochs=5, batch_size=32)
    lstm_predictions = (lstm_model.predict(X_val_pad) > 0.5).astype("int32")
    print("LSTM Classification Report:")
    print(classification_report(y_val, lstm_predictions, zero_division=0))

    return nb_model, svm_model, rf_model, lstm_model, tfidf_vectorizer, tokenizer, max_len, label_encoder

# --- Image Model Selection and Training ---

def extract_image_features(image_tensors, model):
    """Extracts features from image tensors using a pre-trained model."""
    model.eval()
    features = []
    with torch.no_grad():
        for tensor in image_tensors:
            if tensor is not None:
                tensor = tensor.unsqueeze(0)  # Add batch dimension
                feature = model(tensor)
                features.append(feature.squeeze().numpy())
            else:
                features.append(None) # Handle missing tensors
    return features

def train_image_models(train_df, val_df):

    """Trains and evaluates different image models."""

    # Load pre-trained models
    resnet_model = models.resnet50(pretrained=True)
    resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1])) # Remove final layer

    vgg_model = models.vgg16(pretrained=True)
    vgg_model = torch.nn.Sequential(*(list(vgg_model.features))) # Extract features
    #vgg_model = torch.nn.Sequential(*(list(vgg_model.children())[:-1])) #NI Extract features

    efficientnet_model = models.efficientnet_b0(pretrained=True)
    efficientnet_model = torch.nn.Sequential(*(list(efficientnet_model.children())[:-1]))

    # Extract image features
    train_resnet_features = extract_image_features(train_df['image_tensor'].tolist(), resnet_model)
    val_resnet_features = extract_image_features(val_df['image_tensor'].tolist(), resnet_model)


    train_vgg_features = extract_image_features(train_df['image_tensor'].tolist(), vgg_model)
    val_vgg_features = extract_image_features(val_df['image_tensor'].tolist(), vgg_model)


    train_efficientnet_features = extract_image_features(train_df['image_tensor'].tolist(), efficientnet_model)
    val_efficientnet_features = extract_image_features(val_df['image_tensor'].tolist(), efficientnet_model)


    # Remove rows with None feature vectors.
    train_df['resnet_features'] = train_resnet_features
    train_df = train_df.dropna(subset=['resnet_features'])
    val_df['resnet_features'] = val_resnet_features
    val_df = val_df.dropna(subset=['resnet_features'])

    train_df['vgg_features'] = train_vgg_features
    train_df = train_df.dropna(subset=['vgg_features'])
    val_df['vgg_features'] = val_vgg_features
    val_df = val_df.dropna(subset=['vgg_features'])

    train_df['efficientnet_features'] = train_efficientnet_features
    train_df = train_df.dropna(subset=['efficientnet_features'])
    val_df['efficientnet_features'] = val_efficientnet_features
    val_df = val_df.dropna(subset=['efficientnet_features'])



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

# --- Sentiment Analysis ---

def analyze_sentiment(text):
    """Analyzes sentiment of the tweet text using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    return vs['compound']

def extract_sentiment_scores(df):
    """Extracts sentiment scores for each tweet."""
    df['sentiment_score'] = df['processed_text'].apply(analyze_sentiment)
    return df


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
    print(df.shape)
    #print(loader.image_paths)