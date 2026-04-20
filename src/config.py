import os

###########################################
#Harvesting configuration 
###########################################

HARVEST_API_URL = "http://fdp.dcc.sib.swiss"

# Harvester behavior settings
# Set to True to process all datasets, False for only recent ones added since yesterday (default workflow)
RELOAD = False

# Set to True to auto-publish, False to keep datasets in draft
PUBLISH = True


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
