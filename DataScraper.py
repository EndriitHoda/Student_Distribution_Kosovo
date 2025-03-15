import requests
import os
import csv

# List of komunas and their index
komunas = [
    "Deqan","Dargash", "Ferizaj", "Fushe Kosove", "Gjakove", "Gjilan", "Gllogovc", "Hani i Elezit", "Istog",
    "Junik", "Kaqanik", "Kamenice", "Kline", "Leposaviq", "Lipjan", "Malisheve", "Mamushe",
    "Mitrovice", "Novoberde", "Obiliq", "Peje", "Podujeve", "Prishtine", "Prizren", "Rahovec",
    "Shterpce", "Shtime", "Skenderaj", "Suhareke", "Viti", "Vushtrri", "Zubin Potok", "Zveqan", "Kllokot"
]

# Initialize the API URL
api_url = "http://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/1 Pupils in public education/edu05.px"  # Replace with the actual API URL

# Prepare the CSV file to write data
csv_filename = "dataset.csv"
fields = ["Komuna", "Viti Akademik", "Niveli Akademik", "Mosha", "Gjinia", "Numri i nxenesve"]


# Function to make the API request and retrieve the data
def fetch_data_for_komuna(komuna_index, gender_value):
    query = {
        "query": [
            {
                "code": "viti",
                "selection": {
                    "filter": "item",
                    "values": ["8"]
                }
            },
            {
                "code": "komuna",
                "selection": {
                    "filter": "item",
                    "values": [str(komuna_index)]
                }
            },
            {
                "code": "gjinia",
                "selection": {
                    "filter": "item",
                    "values": [str(gender_value)]
                }
            }
        ],
        "response": {
            "format": "json"
        }
    }

    # Send the request to the API
    response = requests.post(api_url, json=query)

    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Failed to fetch data for komuna index {komuna_index}, gender {gender_value}. HTTP Status Code: {response.status_code}")
        return None


# Function to append data to CSV (without overwriting the headers)
def append_to_csv(data):
    # Check if the file already exists to avoid writing headers again
    file_exists = os.path.exists(csv_filename)

    with open(csv_filename, mode='a', newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(fields)

        writer.writerows(data)


# Loop through all komunas and genders
data_rows = []
for index, komuna in enumerate(komunas):
    komuna_index = index  # Komuna value starts from 0 for Dargash

    for gender_value in [0, 1]:  # 0 for Mashkull, 1 for Femer
        gender = "Mashkull" if gender_value == 0 else "Femer"
        response_data = fetch_data_for_komuna(komuna_index, gender_value)

        if response_data:
            # Extract the number of students from the API response
            student_data = response_data.get("data", [])
            if student_data:
                student_count = student_data[0]["values"][0] if "values" in student_data[0] else "0"
            else:
                student_count = "0"  # Default to 0 if no data is available

            # Prepare the row for this komuna and gender
            row = [
                komuna,
                "2023-2024",
                "Institucionet publike parauniversitare",
                "18-19",
                gender,
                student_count
            ]

            data_rows.append(row)


append_to_csv(data_rows)

print(f"New data successfully appended to {csv_filename}")
