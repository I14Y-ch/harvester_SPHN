import os

###########################################
#Harvesting configuration API or FILE 
###########################################

HARVEST_API_URL = "http://fdp.dcc.sib.swiss"

###########################################
# I14Y API configuration
###########################################

# API_BASE_URL = "https://api.i14y.admin.ch/api/partner/v1/" # Prod environement 
API_BASE_URL = "https://api-a.i14y.admin.ch/api/partner/v1/" # ABN enironement 
ACCESS_TOKEN = f"Bearer {os.environ['ACCESS_TOKEN']}" 

# Organization settings
ORGANIZATION_ID = "CH_SPHN"
DEFAULT_PUBLISHER = {
    "identifier": ORGANIZATION_ID
}

