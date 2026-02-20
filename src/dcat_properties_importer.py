from config import *
import namespace
from utils import *
from mappings import *
from rdflib import URIRef, Literal, Graph
from rdflib.namespace import DCTERMS, FOAF, RDFS, DCAT, RDF, SKOS
from rdflib import Namespace
from urllib.parse import urlparse
from typing import Optional, List, Dict, Set
import re

# Constants
SUPPORTED_LANGUAGES = ["de", "en", "fr", "it", "rm"]
DEFAULT_TITLE = {'de': 'Datenexport'}
DEFAULT_DESCRIPTION = {'de': 'Export der Daten'}
DEFAULT_THEME_CODE = '114'


def extract_dataset(graph: Graph, dataset_uri: URIRef) -> Optional[Dict]:
    """
    Extracts dataset details from RDF graph.
    
    Args:
        graph: RDFLib Graph object
        dataset_uri: URI of the dataset
    
    Returns:
        Dictionary with dataset data or None if invalid
    """
    dataset = { 
        "identifiers": [get_literal(graph, dataset_uri, DCTERMS.identifier).split("/")[-1]], # take last part of identifier nly due to identifier in URL
        "title": {"en": get_literal(graph, dataset_uri, DCTERMS.title)},
        "description": {"en": get_literal(graph, URIRef(dataset_uri), DCTERMS.description)},
        "accessRights": get_accessRights(graph, URIRef(dataset_uri), DCTERMS.accessRights),  
        "rights": get_literal(graph, URIRef(dataset_uri), namespace.SPHN_METACAT.hasDataUseRestriction),
        "issued": get_literal(graph, dataset_uri, DCTERMS.issued, is_date=True),
        "modified": get_literal(graph, dataset_uri, DCTERMS.modified, is_date=True),
        "publisher": DEFAULT_PUBLISHER, 
        "landingPages": get_resource_list(graph, dataset_uri, DCAT.landingPage),
        "keywords": get_multilingual_keywords(graph, dataset_uri, DCAT.keyword),
        "languages": get_languages(graph, dataset_uri, DCTERMS.language),
        "contactPoints": extract_contact_points(graph, dataset_uri),
        "documentation": get_resource_list(graph, dataset_uri, FOAF.page),
        "images": get_resource_list(graph, dataset_uri, namespace.SCHEMA.image),
        "temporalCoverage": get_temporal_coverage(graph, dataset_uri), 
        "frequency": get_frequency(graph, dataset_uri),
        "isReferencedBy": get_is_referenced_by(graph, dataset_uri),
        "qualifiedRelations": [ 
                {
                    "hadRole": {
                        "code": "original"
                        },
                    "relation": {
                        "uri": str(dataset_uri)
                    }
                }
            ],
        "relations": get_relations(graph, dataset_uri),
        "spatial": get_spatial(graph, dataset_uri),
        "version": get_literal(graph, dataset_uri, namespace.dcat3.version),
        "versionNotes": get_literal(graph, dataset_uri, namespace.ADMS.versionNotes),
        "conformsTo": get_conforms_to(graph, dataset_uri),
        "themes": get_themes(graph, dataset_uri, DCAT.theme), 
    }

    qualifiedAttributions = get_data_provider_as_qualified_attributions(graph, dataset_uri)
    if qualifiedAttributions:
        dataset["qualifiedAttributions"] = qualifiedAttributions

    if not dataset["description"]:
        print("no description found")
        return None

    return remove_empty_fields(dataset)

def extract_distribution(graph: Graph, distribution_uri: URIRef) -> Dict:
    """
    Extracts a single distribution from a distribution graph.
    
    Args:
        graph: RDFLib Graph object containing the distribution
        distribution_uri: URI of the distribution
        
    Returns:
        Dictionary with distribution data
    """
    title = get_literal(graph, distribution_uri, DCTERMS.title)
    description = get_literal(graph, distribution_uri, DCTERMS.description)
    media_type_uri = get_single_resource(graph, distribution_uri, DCAT.mediaType)
    format_uri = get_single_resource(graph, distribution_uri, DCTERMS.format)
    format_code = None
    if format_uri is not None:
        format_uri_str = str(format_uri)
        format_code = FORMAT_TYPE_MAPPING.get(format_uri_str, format_uri_str.split("/")[-1].upper())

    if title and not description:
        description = f"{title} in {format_code} format"

    download_url = get_single_resource(graph, distribution_uri, DCAT.downloadURL)
    access_url = get_literal(graph, distribution_uri, DCTERMS.identifier) # no access url in original file
    common_url = access_url if access_url is not None else download_url
    download_title = get_literal(graph, distribution_uri, RDFS.label)
    
    availability_uri = get_single_resource(graph, distribution_uri, URIRef("http://data.europa.eu/r5r/availability"))
    license_uri = get_single_resource(graph, distribution_uri, DCTERMS.license)
    license_code = license_uri.split("/")[-1] if license_uri is not None else None
    valid_license = license_code if license_code in VALID_LICENSE_CODES else None
    
    checksum_algorithm = get_literal(graph, distribution_uri, namespace.SPDX.checksumAlgorithm)
    checksum_value = get_literal(graph, distribution_uri, namespace.SPDX.checksumValue)
    packaging_format = get_literal(graph, distribution_uri, DCAT.packageFormat)

    distribution = {
        "title": {"en": title} if title else DEFAULT_TITLE,
        "description": {"en": description} if description else DEFAULT_DESCRIPTION,
        "format": {"code": format_code} if format_code and format_code in VALID_FORMAT_CODES else None,  
        "downloadUrl": {
           # "label": download_title,  
            "uri": download_url if download_url is not None else common_url
        } if common_url is not None else None,
        # "mediaType": {"code": get_media_type(media_type_uri)} if media_type_uri and get_media_type(media_type_uri) else None,
        "accessUrl": {
           # "label": download_title,  
            "uri": access_url 
        } if common_url is not None else None,
        "license": {"code": valid_license} if valid_license is not None else None,  
        "availability": {"code": get_availability_code(availability_uri)} if availability_uri is not None else None,  
        "issued": get_literal(graph, distribution_uri, DCTERMS.issued, is_date=True),
        "modified": get_literal(graph, distribution_uri, DCTERMS.modified, is_date=True),
        "rights": get_literal(graph, distribution_uri, DCTERMS.rights),
        "accessServices": get_access_services(graph, distribution_uri),
        "byteSize": get_literal(graph, distribution_uri, DCAT.byteSize),
        "checksum": {
            "algorithm": {"code": checksum_algorithm} if checksum_algorithm is not None else None,
            "checksumValue": checksum_value
        } if checksum_algorithm is not None or checksum_value is not None else None,
        "conformsTo": get_conforms_to(graph, distribution_uri),
        "coverage": get_coverage(graph, distribution_uri),
        "documentation": get_resource_list(graph, distribution_uri, FOAF.page),
        "identifier": get_literal(graph, distribution_uri, DCTERMS.identifier).split("/")[-1],
        "images": get_resource_list(graph, distribution_uri, namespace.SCHEMA.image),
        "languages": get_languages(graph, distribution_uri, DCTERMS.language),
        "packagingFormat": {"code": packaging_format} if packaging_format is not None else None,
        "spatialResolution": get_literal(graph, distribution_uri, DCAT.spatialResolutionInMeters), 
        "temporalResolution": get_literal(graph, distribution_uri, DCAT.temporalResolution)
    }

    return remove_empty_fields(distribution)


def get_accessRights(graph: Graph, subject: URIRef, predicate: URIRef) -> List[Dict]:
    """Retrieves the access rights field code."""
    for code in graph.objects(subject, predicate):
        if str(code) in VOCAB_EU_ACCESSRIGHTS.keys():
            return {"code": str(code).split("/")[-1]}
    return {"code": "RESTRICTED"}  # Default value if not found

def get_languages(graph: Graph, subject: URIRef, predicate: URIRef) -> List[Dict]:
    """Retrieves a list of i14y codes for languages."""
    return [
        {"code": code}
        for lang_uri in graph.objects(subject, predicate)
        for code, uris in LANGUAGES_MAPPING.items()
        if str(lang_uri) in uris
    ]

def get_multilingual_literal(graph: Graph, subject: URIRef, predicate: URIRef) -> Dict[str, str]:
    """Retrieves multilingual literals from RDF graph."""
    values = {lang: "" for lang in SUPPORTED_LANGUAGES}
    for obj in graph.objects(subject, predicate):
        if isinstance(obj, Literal) and obj.language in values:
            values[obj.language] = remove_html_tags(str(obj))
    return {lang: value for lang, value in values.items() if value}

def get_literal(
    graph: Graph, 
    subject: URIRef, 
    predicate: URIRef, 
    is_date: bool = False
) -> Optional[str]:
    """Retrieves a single value from the RDF graph."""
    value = graph.value(subject, predicate)
    if value is None:
        return None

    value_str = str(value)
    return format_date(value_str) if is_date else value_str

def get_single_resource(graph: Graph, subject: URIRef, predicate: URIRef) -> Optional[str]:
    """Retrieves a single resource (URI) for a given predicate."""
    uri = graph.value(subject, predicate)
    return normalize_uri(str(uri)) if uri is not None else None

def get_resource_list(graph: Graph, subject: URIRef, predicate: URIRef) -> List[Dict]:
    """Retrieves a list of resources (URIs) for a given predicate."""
    return [{"uri": normalize_uri(str(uri))} for uri in graph.objects(subject, predicate)]

def get_multilingual_keywords(graph: Graph, subject: URIRef, predicate: URIRef) -> List[Dict]:
    """Retrieves only keywords with explicit language tags."""
    return [
        {str(lang): str(keyword_obj)}
        for keyword_obj in graph.objects(subject, predicate)
        if keyword_obj is not None and (lang := getattr(keyword_obj, 'language', None))
    ]

# ... 
def get_media_type(media_type_uri: Optional[str]) -> Optional[str]:
    """Returns the media type code if it's a valid URI or direct code."""
    if media_type_uri is None:
        return None
    if media_type_uri in MEDIA_TYPE_MAPPING.values():
        return media_type_uri
    return MEDIA_TYPE_MAPPING.get(str(media_type_uri))

def get_access_services(graph: Graph, subject: URIRef) -> List[Dict]:
    """Retrieves accessServices from RDF graph."""
    return [
        {"id": normalize_uri(str(obj))} 
        for obj in graph.objects(subject, DCAT.accessService)
    ]

def get_coverage(graph: Graph, subject: URIRef) -> List[Dict]:
    """Retrieves coverage from RDF graph."""
    return [
        {"start": start, "end": end}
        for obj in graph.objects(subject, DCTERMS.coverage)
        if (start := get_literal(graph, obj, DCTERMS.start)) is not None or 
           (end := get_literal(graph, obj, DCTERMS.end)) is not None
    ]

def get_spatial(graph: Graph, dataset_uri: URIRef) -> List[str]:
    """Retrieves spatial values as a list of strings."""
    return [
        str(spatial).split("/")[-1] if isinstance(spatial, URIRef) else str(spatial)
        for spatial in graph.objects(dataset_uri, DCTERMS.spatial)
    ] or []

def get_frequency(graph: Graph, subject: URIRef) -> Optional[Dict]:
    """Retrieves frequency from RDF graph."""
    frequency_uri = get_single_resource(graph, subject, DCTERMS.accrualPeriodicity)
    return {"code": frequency_uri.split("/")[-1]} if frequency_uri is not None else None

def get_themes(graph: Graph, subject: URIRef, predicate: URIRef) -> List[Dict]:
    """Retrieves unique theme codes from RDF graph."""
    unique_codes: Set[str] = set()
    themes: List[Dict] = []
    
    for theme in graph.objects(subject, predicate):
        theme_codes = set()
        
        if isinstance(theme, Literal):
            theme_codes.add(str(theme))
        else:
            pref_label = next(graph.objects(theme, SKOS.prefLabel), None)
            if pref_label is not None:
                theme_label = str(pref_label)
                theme_codes.update(
                    code for code, labels in THEME_MAPPING.items()
                    if theme_label in labels
                )
        
        for code in theme_codes:
            if code not in unique_codes:
                unique_codes.add(code)
                themes.append({"code": code})
    if not themes:
        themes.append({"code": DEFAULT_THEME_CODE})
    return themes

def get_availability_code(availability_uri: Optional[str]) -> Optional[str]:
    """Maps availability URI to corresponding code."""
    if availability_uri is None:
        return None
    return next(
        (code for code, uris in VOCAB_EU_PLANNED_AVAILABILITY.items() 
         if availability_uri in uris),
        None
    )

def get_temporal_coverage(graph: Graph, subject: URIRef) -> List[Dict]:
    """Retrieves temporal coverage data from RDF graph."""
    result = []
    
    for obj in graph.objects(subject, DCTERMS.temporal):
        if (obj, RDF.type, URIRef("http://purl.org/dc/terms/PeriodOfTime")) in graph:
            start = get_literal(graph, obj, DCAT.startDate, is_date=True)
            end = get_literal(graph, obj, DCAT.endDate, is_date=True)
            
            if start is not None or end is not None:
                result.append({
                    "start": start,
                    "end": end
                })
                
    return result

def get_is_referenced_by(graph: Graph, subject: URIRef) -> List[Dict]:
    """Retrieves isReferencedBy from RDF graph."""
    return [
        {"uri": normalize_uri(str(obj))}
        for obj in graph.objects(subject, DCTERMS.isReferencedBy)
    ]


def get_data_provider_as_qualified_attributions(graph: Graph, subject: URIRef) -> List[Dict]:
    """Retrieves data providers as qualified attributions from RDF graph."""
    attributions = []
    for obj in graph.objects(subject, namespace.SPHN_METACAT.hasDataProvider):
        uid_url = get_single_resource(graph, obj, DCTERMS.identifier)
        
        # Extract CHE identifier from URL (e.g., CHE-108.910.225)
        identifier = None
        if uid_url:
            match = re.search(r'(CHE[-.]?\d{3}[-.]?\d{3}[-.]?\d{3})', uid_url)
            if match:
                identifier = match.group(1)
        
        # Skip providers without valid identifier
        if not identifier:
            continue
            
        attribution = {
            "agent": {
                "identifier": identifier
            },
            "hadRole": {
                "code": "resourceProvider"
            }
        }
        attributions.append(attribution)
    return attributions

def get_relations(graph: Graph, subject: URIRef) -> List[Dict]:
    """Retrieves relations from RDF graph."""
    relations = []
    for obj in graph.objects(subject, DCTERMS.relation):
        for uri in re.split(r';\s+', str(obj).strip('; \t\n\r')):
            uri = uri.strip()
            if not uri:
                continue
            if is_valid_uri(uri):
                relations.append({
                    "label": get_multilingual_literal(graph, obj, RDFS.label),
                    "uri": normalize_uri(uri)
                })
            else:
                print(f"Skipping invalid relation URI: {uri}")
    return relations

def get_conforms_to(graph: Graph, subject: URIRef) -> List[Dict]:
    """Retrieves conformsTo from RDF graph."""
    return [
        {
            "label": get_multilingual_literal(graph, obj, RDFS.label),
            "uri": normalize_uri(str(obj))
        }
        for obj in graph.objects(subject, DCTERMS.conformsTo)
    ]

def extract_contact_points(graph: Graph, dataset_uri: URIRef) -> List[Dict]:
    """Extracts contact points from RDF."""
    DEFAULT_LANGUAGES = {lang: "" for lang in SUPPORTED_LANGUAGES}
    
    contact_points = []
    for contact_uri in graph.objects(dataset_uri, DCAT.contactPoint):
        fn = str(graph.value(contact_uri, namespace.VCARD.fn)) or get_multilingual_literal(graph, contact_uri, namespace.VCARD.fn)
        email = str(graph.value(contact_uri, namespace.VCARD.hasEmail) or "").removeprefix("mailto:")
        address = get_multilingual_literal(graph, contact_uri, namespace.VCARD.hasAddress)
        telephone = get_literal(graph, contact_uri, namespace.VCARD.hasTelephone)
        note = get_multilingual_literal(graph, contact_uri, namespace.VCARD.note)
        
        if any([fn, email, address, telephone, note]):
            contact_points.append({
                "fn": {"de": fn} if fn else DEFAULT_LANGUAGES,
                "hasAddress": {"de": address} if address else DEFAULT_LANGUAGES,
                "hasEmail": email,
                "hasTelephone": telephone,
                "kind": "Organization",
                "note": note if note else DEFAULT_LANGUAGES
            })
    return contact_points
