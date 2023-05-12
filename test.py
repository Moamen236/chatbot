import requests
import json
import mysql.connector
from datetime import date

# current_year = date.today().year
# print(current_year)

url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
headers = {"x-app-id": "acf12833", "x-app-key": "6e7b86d785c31d0a67ba243a543d8e26", "Content-Type": "application/json"}
query = "1 cup chicken noodle soup, cookies, rice"
data = {"query": query, "timezone": "US/Eastern"}

response = requests.post(url, headers=headers, json=data)
results = response.json()['foods']

# with open('test.txt','w') as fd:
#     fd.write(json.dumps(results,indent=2))

# Filter the results based on the user's goals
filtered_results = []
for result in results:
    if result["nf_calories"] <= 500:
        # filtered_results.append(result)
        print(result["nf_calories"]) 

# Format the results as a string
meal_plans = ""
for result in filtered_results:
    name = result["food_name"]
    calories = result["nf_calories"]
    meal_plans += "{}: {} calories\n".format(name, calories)

# Return the meal plans
print( "Here are some meal plans that might work for you:\n{}".format(meal_plans))


# # Connect to the database
# database = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="",
#     database="gym_system"
# )
# # Create a cursor object to execute SQL queries
# databaseCursor = database.cursor(dictionary=True)

# def executeSql(query, val = None):
#     value = (val)
#     databaseCursor.execute(query, value)
#     results = databaseCursor.fetchall()
#     return results

# typesQuery = "SELECT * FROM types"
# databaseCursor.execute(typesQuery)
# typesResult = databaseCursor.fetchall()
# print("Bot : Choose which type you need (answer by number) :-")
# for type in typesResult:
#     print(type['id'], ": ",type['name'])


# classes = "SELECT * FROM classes WHERE type_id = %s"
# val = (int(1))
# classesResult = executeSql(classes, val)
# print("all Classes", classesResult)
# for oneClass in classesResult:
#     print(oneClass["name"])



