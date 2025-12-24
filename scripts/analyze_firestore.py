import argparse
import pandas as pd
from google.cloud import firestore
from datetime import datetime
import matplotlib.pyplot as plt

def analyze_predictions(project_id, collection='predictions', limit=50, count_only=False):
    """
    Fetches recent predictions from Firestore and displays them.
    """
    print(f"🔥 Connecting to Firestore (Project: {project_id})...")
    db = firestore.Client(project=project_id)
    
    if count_only:
        print(f"🔢 Counting documents in '{collection}'...")
        try:
            # Use aggregation query for efficiency
            count_query = db.collection(collection).count()
            results = count_query.get()
            count = results[0][0].value
            print(f"✅ Total documents in '{collection}': {count}")
        except Exception as e:
            print(f"⚠️  Could not use aggregation query: {e}")
            # Fallback to stream (slow for large collections)
            docs = db.collection(collection).stream()
            count = sum(1 for _ in docs)
            print(f"✅ Total documents in '{collection}': {count}")
        return

    # Query the collection
    print(f"🔍 Fetching last {limit} documents from '{collection}'...")
    docs_stream = db.collection(collection)\
        .order_by('timestamp', direction=firestore.Query.DESCENDING)\
        .limit(limit)\
        .stream()
    
    data = []
    for doc in docs_stream:
        d = doc.to_dict()
        # Flatten for display
        row = {
            'doc_id': doc.id,
            'time': d.get('timestamp_str'),
            'route': d.get('route_id'),
            'stop': d.get('stop_id'),
            'pred_mbt': d.get('predicted_mbt'),
            'status': d.get('status'),
            'created': d.get('created_at')
        }
        data.append(row)
    
    if not data:
        print("❌ No documents found. Is the pipeline running and writing?")
        return

    # Create DataFrame
    df = pd.DataFrame(data)
    
    print("\n📊 Recent Predictions:")
    print(df[['time', 'route', 'stop', 'pred_mbt', 'status']].to_string(index=False))
    
    print("\n📈 Statistics:")
    print(df['pred_mbt'].describe())

    # Optional: Plot
    try:
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        
        plt.figure(figsize=(10, 5))
        plt.plot(df['time'], df['pred_mbt'], marker='o', linestyle='-')
        plt.title(f"Predicted MBT over Time (Last {limit} points)")
        plt.xlabel("Time")
        plt.ylabel("Minutes Between Trains")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_file = "recent_predictions.png"
        plt.savefig(output_file)
        print(f"\n🖼️  Plot saved to {output_file}")
    except Exception as e:
        print(f"Could not plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--limit', type=int, default=50, help='Number of records to fetch')
    parser.add_argument('--count-only', action='store_true', help='Only count total documents')
    args = parser.parse_args()
    
    analyze_predictions(args.project_id, limit=args.limit, count_only=args.count_only)
