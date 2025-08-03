#!/usr/bin/env python3
"""
Data preparation script for DCSC demo
Creates sample data in the expected format for testing
"""

import os
import pickle
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def create_sample_embeddings(texts, embedding_dim=768):
    """
    Create sample embeddings from text using TF-IDF and padding/truncating to match BERT dimensions
    In a real scenario, you would use pre-trained BERT embeddings
    """
    # Use TF-IDF as a simple baseline (would be BERT in real scenario)
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts).toarray()
    except:
        # Fallback for empty or invalid texts
        tfidf_matrix = np.random.random((len(texts), 500))
    
    # Pad or truncate to embedding_dim
    if tfidf_matrix.shape[1] < embedding_dim:
        # Pad with zeros
        padding = np.zeros((tfidf_matrix.shape[0], embedding_dim - tfidf_matrix.shape[1]))
        embeddings = np.concatenate([tfidf_matrix, padding], axis=1)
    else:
        # Truncate
        embeddings = tfidf_matrix[:, :embedding_dim]
    
    return embeddings.astype(np.float32)

def create_sample_data():
    """
    Create sample fake news data for demonstration
    """
    # Sample fake news texts (simplified examples)
    fake_news_samples = [
        "BREAKING: Scientists discover that vaccines contain microchips for mind control! Share this before it gets deleted!",
        "URGENT: Government hiding truth about aliens landing in Area 51 yesterday! Military sources confirm!",
        "SHOCKING: Drinking coffee every day will kill you in 3 months according to secret study!",
        "ALERT: New law will ban all social media next week! Save your photos now!",
        "CONFIRMED: Celebrity XYZ died in car crash this morning but media covering it up!"
    ]
    
    # Sample real news texts
    real_news_samples = [
        "The stock market closed higher today following positive economic indicators released by the Federal Reserve.",
        "Local university announces new research program in renewable energy technology with federal funding.",
        "City council approved budget for new public transportation system set to begin construction next year.",
        "Weather forecast indicates possible snow this weekend with temperatures dropping below freezing.",
        "Technology company reports quarterly earnings above analyst expectations due to strong software sales."
    ]
    
    return fake_news_samples, real_news_samples

def prepare_dataset_files(base_dir, dataset_name='4charliehebdo', num_posts=100):
    """
    Prepare dataset files in the expected pickle format
    """
    fake_texts, real_texts = create_sample_data()
    
    # Create directory structure
    dataset_dir = os.path.join(base_dir, 'data_withdomain', dataset_name)
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    os.makedirs(os.path.join(train_dir, 'rumor'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'nonrumor'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'rumor'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'nonrumor'), exist_ok=True)
    
    # Generate more samples by repeating and slightly modifying
    extended_fake = fake_texts * 10  # Repeat to get more samples
    extended_real = real_texts * 10
    
    # Create embeddings
    all_texts = extended_fake + extended_real
    embeddings = create_sample_embeddings(all_texts)
    
    fake_embeddings = embeddings[:len(extended_fake)]
    real_embeddings = embeddings[len(extended_fake):]
    
    domain_label = dataset_name.replace('4', '')  # Remove prefix number
    
    # Split into train/val (80/20)
    fake_train_size = int(0.8 * len(fake_embeddings))
    real_train_size = int(0.8 * len(real_embeddings))
    
    def save_samples(embeddings_list, labels, domain, save_dir, prefix):
        """Save individual samples as pickle files"""
        for i, (embedding, label) in enumerate(zip(embeddings_list, labels)):
            # Reshape to (N, 768) where N is number of posts
            # For demo, we'll just use 1 post per sample
            X = embedding.reshape(1, -1)
            
            # Ensure we don't exceed num_posts
            if X.shape[0] > num_posts:
                X = X[:num_posts]
            
            # Create sample tuple: (X, y, domain_label)
            sample_data = (X, label, domain)
            
            filename = f"{prefix}_{i:04d}.pkl"
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(sample_data, f)
    
    # Save training data
    print(f"📁 Creating training data...")
    save_samples(
        fake_embeddings[:fake_train_size], 
        [1] * fake_train_size,  # 1 for rumor/fake
        domain_label,
        os.path.join(train_dir, 'rumor'),
        'fake'
    )
    
    save_samples(
        real_embeddings[:real_train_size],
        [0] * real_train_size,  # 0 for non-rumor/real  
        domain_label,
        os.path.join(train_dir, 'nonrumor'),
        'real'
    )
    
    # Save validation data
    print(f"📁 Creating validation data...")
    save_samples(
        fake_embeddings[fake_train_size:],
        [1] * (len(fake_embeddings) - fake_train_size),
        domain_label,
        os.path.join(val_dir, 'rumor'),
        'fake'
    )
    
    save_samples(
        real_embeddings[real_train_size:],
        [0] * (len(real_embeddings) - real_train_size),
        domain_label,
        os.path.join(val_dir, 'nonrumor'),
        'real'
    )
    
    # Create target domain data directory
    target_dir = os.path.join(base_dir, 'raw_data_withdomain')
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"✅ Dataset '{dataset_name}' created successfully!")
    print(f"   📊 Training: {fake_train_size} fake + {real_train_size} real")
    print(f"   📊 Validation: {len(fake_embeddings) - fake_train_size} fake + {len(real_embeddings) - real_train_size} real")
    print(f"   📁 Location: {dataset_dir}")

def main():
    """Main function to set up demo data"""
    print("🔧 Preparing DCSC demo data...")
    print("=" * 40)
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    demo_data_dir = os.path.join(current_dir, 'demo_data')
    
    # Create demo data for multiple domains
    datasets = ['4charliehebdo', '4ferguson', '4germanwings-crash']
    
    for dataset in datasets:
        print(f"\n📦 Creating dataset: {dataset}")
        prepare_dataset_files(demo_data_dir, dataset)
    
    print(f"\n🎉 Demo data preparation completed!")
    print(f"📁 Data location: {demo_data_dir}")
    print(f"\n▶️  Run demo with: python demo_main.py")

if __name__ == '__main__':
    main()