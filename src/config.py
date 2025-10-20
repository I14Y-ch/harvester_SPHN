import os

###########################################
#Harvesting configuration API or FILE 
###########################################

HARVEST_API_URL = "http://fdp.dcc.sib.swiss"

###########################################
# I14Y API configuration
###########################################

API_BASE_URL = os.environ['API_BASE_URL'] 
ACCESS_TOKEN = f"Bearer {os.environ['ACCESS_TOKEN']}"

# Organization settings
ORGANIZATION_ID = "CH_SPHN"
DEFAULT_PUBLISHER = {
    "identifier": ORGANIZATION_ID
}

