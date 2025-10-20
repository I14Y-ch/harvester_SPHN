import sys
import json
import requests
import time
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import DCTERMS, DCAT, RDF
import argparse
import urllib3
import datetime
from config import *

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define additional namespaces
LDP = Namespace("http://www.w3.org/ns/ldp#")
FDP = Namespace("https://w3id.org/fdp/fdp-o#")

def fetch_rdf(url):
    """
    Fetch RDF data from a URL without saving to file.
    
    Args:
        url: URL to fetch RDF data from
    
    Returns:
        Tuple of (rdf_data, format) if successful, (None, None) otherwise
    """
    print(f"Fetching RDF from {url}")
    try:
        session = requests.Session()
        
        # Fetch the RDF data
        response = session.get(
            url=url,
            verify=False,
            timeout=40.0,
            headers={
                'Accept': 'text/turtle, application/rdf+xml, */*;q=0.5'
            }
        )
        
        response.raise_for_status()
        
        # Determine format from content type
        content_type = response.headers.get('Content-Type', '')
        if 'turtle' in content_type:
            detected_format = 'turtle'
        elif 'rdf+xml' in content_type or 'xml' in content_type:
            detected_format = 'xml'
        else:
            detected_format = 'turtle'  # Default to turtle
        
        return response.text, detected_format
    except requests.RequestException as e:
        print(f"Error fetching RDF from {url}: {e}")
        return None, None

def extract_catalog_uris(fdp_graph):
    """
    Extract catalog URIs from a FAIR Data Point RDF graph.
    """
    catalog_uris = []
    
    # Method 1: Get catalogs from ldp:contains relationships
    for s, p, o in fdp_graph.triples((None, LDP.contains, None)):
        # Check if object is a catalog
        if (o, RDF.type, DCAT.Catalog) in fdp_graph:
            catalog_uris.append(str(o))
        else:
            # Add anyway, we'll check when we fetch it
            catalog_uris.append(str(o))
    
    # Method 2: Get catalogs from metadataCatalog relationships
    for s, p, o in fdp_graph.triples((None, FDP.metadataCatalog, None)):
        catalog_uri = str(o)
        if catalog_uri not in catalog_uris:
            catalog_uris.append(catalog_uri)
    
    return catalog_uris

def extract_dataset_uris(catalog_graph):
    """
    Extract dataset URIs from a catalog RDF graph.
    """
    dataset_uris = []
    
    # Look for dataset references in the catalog
    for s, p, o in catalog_graph.triples((None, DCAT.dataset, None)):
        dataset_uris.append(str(o))
    
    # Also look for ldp:contains relationships
    for s, p, o in catalog_graph.triples((None, LDP.contains, None)):
        # Check if object is a dataset
        if (o, RDF.type, DCAT.Dataset) in catalog_graph:
            dataset_uri = str(o)
            if dataset_uri not in dataset_uris:
                dataset_uris.append(dataset_uri)
    
    return dataset_uris

def extract_dataset_id(dataset_uri):
    """
    Extract dataset ID from a dataset URI.
    """
    # Parse the URI and get the last part
    parts = dataset_uri.rstrip('/').split('/')
    return parts[-1]

def fetch_dataset(dataset_id):
    """
    Fetch dataset RDF and convert to structured data.
    """
    dataset_url = f"{HARVEST_API_URL}/dataset/{dataset_id}"
    dataset_rdf, rdf_format = fetch_rdf(dataset_url)
    
    if not dataset_rdf:
        print(f"Failed to retrieve dataset RDF for {dataset_id}")
        return None, None, None, None
    
    # Parse dataset RDF
    dataset_graph = Graph()
    try:
        dataset_graph.parse(data=dataset_rdf, format=rdf_format)
        print(f"Parsed dataset graph with {len(dataset_graph)} triples")
        
        # Find dataset URI in the graph
        dataset_uris = list(dataset_graph.subjects(
            predicate=RDF.type, 
            object=DCAT.Dataset
        ))
        
        if not dataset_uris:
            print(f"No dataset found in RDF for {dataset_id}")
            return None, None, None, None
        
        # Extract dataset URI
        dataset_uri = dataset_uris[0]
        
        # Extract metadataIssued and metadataModified timestamps
        metadata_issued = None
        metadata_modified = None
        
        for issued in dataset_graph.objects(dataset_uri, FDP.metadataIssued):
            metadata_issued = str(issued)
            break
            
        for modified in dataset_graph.objects(dataset_uri, FDP.metadataModified):
            metadata_modified = str(modified)
            break
        
        # Extract dataset data using your existing function
        from dcat_properties_importer import extract_dataset
        dataset_data = extract_dataset(dataset_graph, dataset_uri)
        
        if not dataset_data:
            print(f"Failed to extract dataset data for {dataset_id}")
            return None, None, None, None
        
        return {"data": dataset_data}, dataset_graph, metadata_issued, metadata_modified
        
    except Exception as e:
        print(f"Error processing dataset {dataset_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def fetch_distribution(distribution_id):
    """
    Fetch distribution RDF and convert to structured data.
    """
    dist_url = f"{HARVEST_API_URL}/distribution/{distribution_id}"
    dist_rdf, rdf_format = fetch_rdf(dist_url)
    
    if not dist_rdf:
        print(f"Failed to retrieve distribution RDF for {distribution_id}")
        return None
    
    # Parse distribution RDF
    dist_graph = Graph()
    try:
        dist_graph.parse(data=dist_rdf, format=rdf_format)
        print(f"Parsed distribution graph with {len(dist_graph)} triples")
        
        # Find distribution URI in the graph
        distribution_uris = list(dist_graph.subjects(
            predicate=RDF.type, 
            object=DCAT.Distribution
        ))
        
        if not distribution_uris:
            print(f"No distribution found in RDF for {distribution_id}")
            return None
        
        # Extract distribution data
        from dcat_properties_importer import extract_distribution
        distribution_uri = distribution_uris[0]
        distribution_data = extract_distribution(dist_graph, distribution_uri)
        
        if not distribution_data:
            print(f"Failed to extract distribution data for {distribution_id}")
            return None

        # Add ID to the distribution data
        # distribution_data["id"] = distribution_id
        
        return distribution_data
        
    except Exception as e:
        print(f"Error processing distribution {distribution_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_distribution_ids(dataset_graph):
    """
    Extract distribution IDs from a dataset graph.
    """
    distribution_ids = []
    
    # Find all distribution URIs in the dataset
    for s, p, o in dataset_graph.triples((None, DCAT.distribution, None)):
        # Extract ID from the URI
        dist_id = str(o).split('/')[-1]
        distribution_ids.append(dist_id)
    
    return distribution_ids

def process_catalog_data(catalog_uri, processed_datasets=None, target_url=None):
    """
    Process a catalog by URI, extracting and processing all datasets.
    """
    if processed_datasets is None:
        processed_datasets = set()
    
    print(f"\n=== Processing catalog: {catalog_uri} ===")
    
    # Extract catalog ID
    catalog_id = extract_dataset_id(catalog_uri)
    
    # Fetch and parse catalog RDF
    catalog_rdf, rdf_format = fetch_rdf(catalog_uri)
    
    if not catalog_rdf:
        print(f"Failed to fetch catalog RDF for {catalog_uri}")
        return {
            "catalog_id": catalog_id,
            "catalog_uri": catalog_uri,
            "success": False,
            "error": "Failed to fetch catalog RDF",
            "datasets": []
        }
    
    # Parse the catalog RDF
    catalog_graph = Graph()
    try:
        catalog_graph.parse(data=catalog_rdf, format=rdf_format)
        print(f"Parsed catalog graph with {len(catalog_graph)} triples")
        
        # Extract catalog metadata
        catalog_title = None
        for title in catalog_graph.objects(URIRef(catalog_uri), DCTERMS.title):
            catalog_title = str(title)
            break
        
        # Extract dataset URIs from the catalog
        dataset_uris = extract_dataset_uris(catalog_graph)
        print(f"Found {len(dataset_uris)} datasets in catalog")
        
        # Process each dataset
        processed_dataset_info = []
        for dataset_uri in dataset_uris:
            dataset_id = extract_dataset_id(dataset_uri)
            
            # Skip if already processed
            if dataset_id in processed_datasets:
                print(f"Dataset {dataset_id} already processed, skipping")
                continue
            
            print(f"\n--- Processing dataset: {dataset_id} ---")
            # Fetch the dataset using our custom function
            dataset_data, dataset_graph, metadata_issued, metadata_modified = fetch_dataset(dataset_id)
            
            # Process distributions if dataset was successfully fetched
            if dataset_data and dataset_graph:
                try:
                    # Extract distribution IDs from the dataset graph
                    distribution_ids = extract_distribution_ids(dataset_graph)
                    print(f"Found {len(distribution_ids)} distributions to process for dataset {dataset_id}")
                    
                    # Process each distribution separately
                    distribution_data_list = []
                    
                    for dist_id in distribution_ids:
                        print(f"Processing distribution ID: {dist_id}")
                        dist_data = fetch_distribution(dist_id)
                        
                        if dist_data:
                            distribution_data_list.append(dist_data)
                    
                    # Add distribution data to dataset
                    if distribution_data_list:
                        dataset_data['data']['distributions'] = distribution_data_list
                        print(f"Added {len(distribution_data_list)} distributions to dataset")
                    
                    # Post to target if URL is provided
                    post_success, action = post_to_i14y(dataset_data, metadata_issued, metadata_modified)
                    print(f"Posted dataset {dataset_id} to i14y: {'Success' if post_success else 'Failed'}")

                    processed_datasets.add(dataset_id)
                    processed_dataset_info.append({
                        "dataset_id": dataset_id,
                        "dataset_uri": dataset_uri,
                        "distributions_count": len(distribution_ids),
                        "processed_distributions": len(distribution_data_list),
                        "action": action,
                        "success": post_success
                    })
                except Exception as e:
                    print(f"Error processing distributions for dataset {dataset_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    processed_dataset_info.append({
                        "dataset_id": dataset_id,
                        "dataset_uri": dataset_uri,
                        "success": True,
                        "distributions_error": str(e)
                    })
            else:
                processed_dataset_info.append({
                    "dataset_id": dataset_id,
                    "dataset_uri": dataset_uri,
                    "success": False,
                    "error": "Failed to process dataset"
                })
        
        # Create catalog summary
        catalog_summary = {
            "catalog_id": catalog_id,
            "catalog_uri": catalog_uri,
            "title": catalog_title,
            "dataset_count": len(dataset_uris),
            "success": True,
            "datasets": processed_dataset_info
        }
        
        return catalog_summary
        
    except Exception as e:
        print(f"Error processing catalog {catalog_uri}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "catalog_id": catalog_id,
            "catalog_uri": catalog_uri,
            "success": False,
            "error": str(e),
            "datasets": []
        }


def post_to_i14y(data, metadata_issued=None, metadata_modified=None):
    """
    Post the data to i14y, with conditional update based on metadata timestamps
    """
    try:
        # Determine whether to create or update based on metadata timestamps
        create_new = False
        update_existing = False
        dataset_id = None
        action = "not_modified"  # Default action
        
        if "data" in data and "id" in data["data"]:
            dataset_id = data["data"]["id"]
        
        if metadata_issued and is_more_recent_than_yesterday(metadata_issued):
            create_new = True
            print(f"Dataset was recently issued ({metadata_issued}), creating new dataset")
            action = "created"
        elif metadata_modified and is_more_recent_than_yesterday(metadata_modified):
            update_existing = True
            print(f"Dataset was recently modified ({metadata_modified}), updating existing dataset")
            action = "updated"
        else:
            print("Dataset hasn't been recently issued or modified, skipping")
            return False, "not_modified"
        
        if create_new:
            # Create new dataset with POST
            response = requests.post(
                url=API_BASE_URL + '/datasets',
                json=data,
                headers={'Authorization': ACCESS_TOKEN, 'Content-Type': 'application/json', 'Accept': '*/*','Accept-encoding': 'json'}, 
                verify=False
            )
            
        elif update_existing and dataset_id:
            # Update existing dataset with PUT
            response = requests.put(
                url=f"{API_BASE_URL}/datasets/{dataset_id}",
                json=data,
                headers={'Authorization': ACCESS_TOKEN, 'Content-Type': 'application/json', 'Accept': '*/*','Accept-encoding': 'json'}, 
                verify=False
            )
        else:
            print("Cannot update: missing dataset ID")
            return False, "error"
        
        response.raise_for_status()
        print(f"Successfully {action} dataset")
        return True, action
    except requests.RequestException as e:
        print(f"Error processing dataset: {e}")
        try:
            print(f"Response: {response.text}")
        except:
            pass
        return False, "error"

def post_all_to_i14y(data, metadata_issued=None, metadata_modified=None):
    """
    Post all datasets to i14y regardless of timestamps.
    Use this function for initial data import only.
    
    Args:
        data: Dataset data to post
        metadata_issued: Ignored for this function
        metadata_modified: Ignored for this function
        
    Returns:
        Tuple of (success_boolean, action_string)
    """
    try:
        dataset_id = None
        if "data" in data and "id" in data["data"]:
            dataset_id = data["data"]["id"]
        
        # Always create new for initial import
        print(f"Initial import: creating dataset")
        
        # Create dataset with POST
        response = requests.post(
            url=API_BASE_URL + '/datasets',
            json=data,
            headers={'Authorization': ACCESS_TOKEN, 'Content-Type': 'application/json', 'Accept': '*/*','Accept-encoding': 'json'}, 
            verify=False
        )

        response.raise_for_status()
        print(f"Successfully created dataset")
        return True, "created"
    except requests.RequestException as e:
        print(f"Error creating dataset: {e}")
        try:
            print(f"Response: {response.text}")
        except:
            pass
        return False, "error"

def is_more_recent_than_yesterday(date_str):
    """
    Check if a date string is more recent than yesterday.
    
    Args:
        date_str: ISO format date string
        
    Returns:
        Boolean indicating if the date is more recent than yesterday
    """
    try:
        # Extract the date part
        date_part = date_str.split('T')[0]
        parsed_date = datetime.date.fromisoformat(date_part)
        
        # Get yesterday's date
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        
        # Compare dates
        return parsed_date > yesterday
    except Exception as e:
        print(f"Error parsing date: {e}")
        return False  # Default to False if we can't parse the date
    

def main():
    
    # Record start time
    start_time = time.time()
    
    # Step 1: Fetch the FDP RDF
    fdp_rdf, rdf_format = fetch_rdf(HARVEST_API_URL)
    
    if not fdp_rdf:
        print("Failed to fetch FDP RDF. Exiting.")
        sys.exit(1)
    
    # Step 2: Parse the FDP RDF
    fdp_graph = Graph()
    try:
        fdp_graph.parse(data=fdp_rdf, format=rdf_format)
        print(f"Parsed FDP graph with {len(fdp_graph)} triples")
        
        # Step 3: Extract catalog URIs
        catalog_uris = extract_catalog_uris(fdp_graph)
        print(f"Found {len(catalog_uris)} catalog URIs")
        
        # Step 4: Process each catalog
        processed_datasets = set()
        catalog_summaries = []
        
        for catalog_uri in catalog_uris:
            summary = process_catalog_data(catalog_uri, processed_datasets)
            catalog_summaries.append(summary)
            
            # Add a small delay to avoid overwhelming the server
            time.sleep(1)
        
        # Step 5: Create overall summary
        harvest_summary = {
            "harvest_date": datetime.datetime.now().isoformat(),
            "catalog_count": len(catalog_uris),
            "dataset_count": len(processed_datasets),
            "duration_seconds": round(time.time() - start_time, 2),
            "catalogs": catalog_summaries
        }
        
        print(f"\n=== Harvest Summary ===")
        print(f"Processed {len(catalog_uris)} catalogs")
        print(f"Processed {len(processed_datasets)} unique datasets")
        print(f"Total time: {harvest_summary['duration_seconds']} seconds")
        
        # Create log to upload as artifact
        try:
            log = f"Harvest completed successfully at {datetime.datetime.now()}\n\n"
            log += f"=== Harvest Summary ===\n"
            log += f"Processed {len(catalog_uris)} catalogs\n"
            log += f"Processed {len(processed_datasets)} unique datasets\n"
            log += f"Total time: {harvest_summary['duration_seconds']} seconds\n\n"
            
            # Add catalog details
            log += "=== Catalogs Processed ===\n"
            for summary in catalog_summaries:
                log += f"Catalog: {summary['catalog_id']}\n"
                if not summary['success'] and 'error' in summary:
                    log += f"  Error: {summary['error']}\n"
                log += "\n"
            
            # Add processed datasets with their status
            log += "=== Datasets Processed ===\n"
            
            # Track counts for each action type
            created_count = 0
            updated_count = 0
            not_modified_count = 0
            error_count = 0
            
            # Process all datasets from all catalogs
            for summary in catalog_summaries:
                if summary['success'] and 'datasets' in summary:
                    for dataset in summary['datasets']:
                        dataset_id = dataset['dataset_id']
                        action = dataset.get('action', 'unknown')
                        
                        # Update counts
                        if action == 'created':
                            created_count += 1
                            status = "CREATED"
                        elif action == 'updated':
                            updated_count += 1
                            status = "UPDATED"
                        elif action == 'not_modified':
                            not_modified_count += 1
                            status = "NOT MODIFIED"
                        else:
                            error_count += 1
                            status = "ERROR"
                        
                        log += f"- {dataset_id}: {status}\n"
            
            # Add action summary
            log += f"\n=== Action Summary ===\n"
            log += f"Created: {created_count}\n"
            log += f"Updated: {updated_count}\n"
            log += f"Not modified: {not_modified_count}\n"
            log += f"Errors: {error_count}\n"

        except Exception as e:
            log = f"Harvest failed at {datetime.datetime.now()}: {str(e)}\n"
            raise
        finally:
            # Save log in root directory
            with open('harvest_log.txt', 'w') as f:
                f.write(log)
                
    except Exception as e:
        print(f"Error processing FDP: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()