from diagrams import Diagram, Cluster, Edge
from diagrams.gcp.compute import ComputeEngine
from diagrams.gcp.analytics import PubSub, Dataflow, BigQuery
from diagrams.gcp.database import Firestore
from diagrams.gcp.ml import AIPlatform

# Attributes for the diagram and clusters to increase font size
main_graph_attr = {
    "fontsize": "45",
    "bgcolor": "white",
    "labelloc": "t",
    "dpi": "300"
}

cluster_attr = {
    "fontsize": "30"
}

edge_attr = {
    "fontsize": "20"
}

# Create the diagram
with Diagram("Prediction Service Architecture\n\n", show=False, direction="LR", graph_attr=main_graph_attr):
    
    # 1. ML Training Pipeline (Offline)
    with Cluster("ML Training Pipeline (Vertex AI)", graph_attr=cluster_attr):
        bq = BigQuery("BigQuery\n(Historical Data)")
        
        with Cluster("Vertex AI Pipeline", graph_attr=cluster_attr):
            preprocess = AIPlatform("Preprocessing")
            
            with Cluster("Parallel Training", graph_attr=cluster_attr):
                train1 = AIPlatform("Train Model A")
                train2 = AIPlatform("Train Model B")
            
            with Cluster("Parallel Evaluation", graph_attr=cluster_attr):
                eval1 = AIPlatform("Eval Model A")
                eval2 = AIPlatform("Eval Model B")
                
            model_spec = AIPlatform("Model Spec")
            registry = AIPlatform("Model Registry")

        # Flow
        bq >> preprocess
        preprocess >> train1 >> eval1
        preprocess >> train2 >> eval2
        [eval1, eval2] >> model_spec >> registry

    # 2. Streaming & Serving (Online)
    with Cluster("Streaming & Serving System", graph_attr=cluster_attr):
        gce = ComputeEngine("GCE Ingestion\n(Polling Script)")
        pubsub = PubSub("Pub/Sub\n(Message Queue)")
        dataflow = Dataflow("Dataflow\n(Enrichment & Pred)")
        endpoint = AIPlatform("Vertex AI\nEndpoint")
        firestore = Firestore("Firestore\n(Results)")

        # Flow
        # We apply edge_attr to specific edges where we want larger text
        gce >> Edge(label="Publish 30s", **edge_attr) >> pubsub >> dataflow
        dataflow >> Edge(label="Predict", **edge_attr) >> endpoint
        dataflow >> Edge(label="Write", **edge_attr) >> firestore
    
    # Connection between Offline and Online
    registry >> Edge(label="Manual Deploy", style="dashed", color="firebrick", **edge_attr) >> endpoint
