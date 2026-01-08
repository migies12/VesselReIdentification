import api_helper
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()
    access_token = api_helper.get_access_token(os.getenv("SKYLIGHT_USERNAME"), os.getenv("SKYLIGHT_PASSWORD"))
    events = api_helper.get_recent_correlated_vessels(access_token, 1)

    print(events)