import requests
import csv
import os

def fetch_data_for_api(api_url, query={
            "query": [],
            "response": {
                "format": "json"
            }
        }):

    response = requests.post(api_url, json=query)
    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Failed to fetch data for api {api_url}. HTTP Status Code: {response.status_code}")
        return None

def fetch_definition_for_api(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Failed to fetch definition for api {api_url}. HTTP Status Code: {response.status_code}")
        return None

def fetch_api_output_to_csv(api_url, csv_file_name, folder="datasets"):
    # Check if the CSV file already exists
    if os.path.exists(folder + "/" + csv_file_name):
        # print(f"File '{csv_file_name}' already exists. Skipping write operation.")
        return

    definition = fetch_definition_for_api(api_url)
    api_response = fetch_data_for_api(api_url)

    #Create mapping from definition
    mappings = {}
    for variable in definition["variables"]:
        mappings[variable["code"]] = dict(zip(variable["values"], variable["valueTexts"]))

    # Extract CSV headers
    headers = [col["text"] for col in api_response["columns"]]

    #Write CSV file
    with open(folder + "/" + csv_file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in api_response["data"]:
            mapped_keys = [mappings[code][key] for code, key in zip(mappings.keys(), row["key"])]
            writer.writerow(mapped_keys + row["values"])

datasets = {
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/2 Educational staff/edu16.px": "dataset_stafi_akademik.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/2 Educational staff/edu19.px": "dataset_stafi_administrativ.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/2 Educational staff/edu21.px": "dataset_stafi_ndihmes.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/5 Learning conditions/edu26.px": "dataset_raportet_nxenes_staf.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/6 Information on social and family status of pupils/edu28.px": "dataset_statusi_social.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Household budget survey/1 Overall consumption in Kosovo/hbs01.px": "dataset_konsumi_i_pergjithshem.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Environment/Uji/tab14.px": "dataset_uji_humbur.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Population/Births and deaths/birthsDeathsHistory.px": "dataset_lindjet_vdekjet.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Population/Estimate, projection and structure of population/Population projection/Population projection, 2011-2061/tab1vMesem.px": "dataset_populsia_with_prediction.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/ICT/ICT in Households/TIK1point1.px": "dataset_qasja_ne_internet.csv"
}

# Iterate through the dictionary
for url, csv_name in datasets.items():
    fetch_api_output_to_csv(url, csv_name)