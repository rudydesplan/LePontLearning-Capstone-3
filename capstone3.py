import os
import json
from azure.storage.blob import BlobServiceClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from unidecode import unidecode

# Initialize credentials and clients
def initialize_clients():
    credential = DefaultAzureCredential()
    vault_url = "https://lepont2.vault.azure.net/"
    client = SecretClient(vault_url=vault_url, credential=credential)
    return client, credential

# Retrieve secrets from Azure Key Vault
def get_secrets(client):
    azure_storage_connection_string = client.get_secret("azurestorageconnection").value
    azure_text_analytics_key = client.get_secret("azuretextanalyticskey").value
    azure_text_analytics_endpoint = client.get_secret("AzureTextAnalyticsEndpoint").value
    container_name1 = client.get_secret("containername1").value
    output_container_name = client.get_secret("outputcontainername").value
    return azure_storage_connection_string, azure_text_analytics_key, azure_text_analytics_endpoint, container_name1, output_container_name

# Download blob content
def download_blob(blob_service_client, container_name, blob_name):
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    stream = blob_client.download_blob().readall().decode('utf-8')
    return stream

# Process and decode tweets
def process_tweets(stream):
    data = [json.loads(line) for line in stream.strip().split('\n') if line]
    for tweet in data:
        tweet['text'] = unidecode(tweet['text'])
    return data

# Upload data to blob
def upload_data(blob_service_client, container_name, blob_name, data):
    output_container_client = blob_service_client.get_container_client(container_name)
    output_blob_client = output_container_client.get_blob_client(blob_name)
    try:
        output_blob_client.upload_blob(data, overwrite=True)
        print("Data uploaded successfully.")
    except Exception as e:
        print(f"Failed to upload data: {str(e)}")

# Analyze sentiment of tweets
def analyze_sentiment(text_analytics_client, texts):
    # Maximum size for each document batch is 10
    max_batch_size = 10
    sentiments = []
    
    # Process batches of up to 10 documents
    for i in range(0, len(texts), max_batch_size):
        batch_texts = texts[i:i + max_batch_size]
        response = text_analytics_client.analyze_sentiment(documents=batch_texts)
        sentiments.extend([(doc.sentiment, doc.confidence_scores) for doc in response])
    
    return sentiments


# Main processing function
def main():
    client, credential = initialize_clients()
    secrets = get_secrets(client)
    blob_service_client = BlobServiceClient.from_connection_string(secrets[0])
    text_analytics_client = TextAnalyticsClient(endpoint=secrets[2], credential=AzureKeyCredential(secrets[1]))
    blob_name = "tweets/2024-05-07_TwitterData.json"

    # Download and process tweets
    stream = download_blob(blob_service_client, secrets[3], blob_name)
    tweets = process_tweets(stream)

    # Serialize processed data
    serialized_data = json.dumps(tweets)
    upload_data(blob_service_client, secrets[4], blob_name, serialized_data)

    # Sentiment analysis
    tweet_texts = [tweet['text'] for tweet in tweets]
    sentiments = analyze_sentiment(text_analytics_client, tweet_texts)
    results = [{
        "TweetText": tweet['text'],
        "Sentiment": sentiment[0],
        "PositiveScore": sentiment[1].positive,
        "NeutralScore": sentiment[1].neutral,
        "NegativeScore": sentiment[1].negative
    } for tweet, sentiment in zip(tweets, sentiments)]

    # Save results to Parquet
    df = pd.DataFrame(results)
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_sentiment_analysis.parquet"
    full_path = os.path.join(os.getcwd(), filename)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, full_path)

    # Upload Parquet file
    upload_data(blob_service_client, "sentimentanalysis", filename, open(full_path, 'rb').read())

if __name__ == "__main__":
    main()