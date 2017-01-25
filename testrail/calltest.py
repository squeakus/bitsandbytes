import testrail
import json

client = testrail.APIClient('https://jonathan.testrail.com/')
client.user = 'jonathanbyrn@gmail.com'
client.password = '0mayZRF3wZkNoEUABF9A'

case = client.send_get('get_case/1')
cleanedcase = json.dumps(case)
print(cleanedcase)


# result = client.send_post(
# 	'add_result_for_case/1/1',
# 	{ 'status_id': 1, 'comment': 'This test worked fine!' }
# )

# result = client.send_post(
# 	'add_result_for_case/1/2',
# 	{ 'status_id': 5, 'comment': 'This test failed miserably' }
# )

# result = client.send_post(
# 	'add_result_for_case/1/3',
# 	{ 'status_id': 5, 'comment': 'Could be worth looking at' }
# )
 
# print(result)
