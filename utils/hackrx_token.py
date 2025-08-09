import requests
from config import TOKEN

def get_team_token():
    url = "https://register.hackrx.in/submissions/getTeamToken"

    headers = {
        "accept": "*/*",
        "origin": "https://dashboard.hackrx.in",
        "referer": "https://dashboard.hackrx.in/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "x-authorization": f"Bearer {TOKEN}"
    }

    response = requests.get(url, headers=headers)

    # print("Status Code:", response.status_code)
    print("Response JSON:", response.json().get("encrypted"))
    return response



if __name__ == "__main__":
    # This will execute when the script is run directly
    get_team_token()